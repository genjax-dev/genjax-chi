# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp

import genjax
import genjax.experimental.prox as prox


#####
# Discrete HMM
#####


@genjax.gen
def kernel_step(key, prev, transition_t, observation_t):
    trow = transition_t[prev, :]
    key, latent = genjax.trace("latent", genjax.Categorical)(key, (trow,))
    orow = observation_t[latent, :]
    key, observation = genjax.trace("observation", genjax.Categorical)(
        key, (orow,)
    )
    return key, latent


kernel = genjax.Unfold(kernel_step, max_length=50)


def initial_position(config: genjax.DiscreteHMMConfiguration):
    return jnp.array(int(config.linear_grid_dim / 2))


@genjax.gen
def hidden_markov_model(key, T, config):
    z0 = initial_position(config)
    transition_t = config.transition_tensor
    observation_t = config.observation_tensor
    key, z = genjax.trace("z", kernel)(
        key, (T, z0, transition_t, observation_t)
    )
    return key, z


#####
# Custom transition proposal
#####

# This generates a new target distribution for the SMC chain.
# See: (SMCExtend) in Prox.
def hmm_meta_next_target(state, constraints, final_target):
    args = final_target.args

    # Allows coercions of proposal
    # address structure to match a step target's
    # structure.
    def choice_map_coercion(chm):
        return chm["z"]

    transition_tensor = args[1].transition_tensor
    observation_tensor = args[1].observation_tensor

    return prox.Target(
        kernel_step,
        choice_map_coercion,
        (state, transition_tensor, observation_tensor),
        constraints["z"],
    )


@genjax.gen(
    prox.ChoiceMapDistribution,
    selection=genjax.AllSelection(),
)
def transition_proposal(key, state, new_target):
    config = new_target.args[1]
    observation = new_target.constraints["observation"]
    trow = new_target.args[1][state, :]
    orow = new_target.args[2][:, observation]
    observation_weights = orow[jnp.arange(0, len(trow))]
    weights = trow + observation_weights
    key, _ = genjax.trace(("z", "latent"), genjax.Categorical)(key, (weights,))
    return (key,)


@genjax.prox.smc.propagator
def algorithm(key, state):
    def _inner(carry, x):
        key, state = carry
        new_args, new_constraints = x
        key, state = prox.smc_step(prox.smc.Extend(q))(
            key, state, new_args, new_constraints
        )
        key, state = prox.smc_step(prox.smc.Resample(0.7, 50))(key, state)
        return (key, state), ()

    key, state = jax.xla.scan(
        _inner,
        (key, state),
        length=50,
    )
    return key, state
