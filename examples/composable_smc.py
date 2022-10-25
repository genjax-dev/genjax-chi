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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import genjax
import genjax.experimental.prox as prox
from genjax.core.masks import IndexMask


console = genjax.go_pretty()

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


kernel = genjax.Unfold(kernel_step, max_length=5)


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
# SMC DSL example
#####


@genjax.gen(
    prox.ChoiceMapDistribution,
    selection=genjax.AllSelection(),
)
def initial_proposal(key, target):
    config = target.args[1]
    state = initial_position(config)
    obs = target.constraints["z", "observation"]
    trow = config.transition_tensor[state, :]
    orow = config.observation_tensor[:, obs]
    observation_weights = orow[jnp.arange(0, len(trow))]
    weights = trow + observation_weights
    key, _ = genjax.trace(("z", "latent"), genjax.Categorical)(key, (weights,))
    return (key,)


@genjax.gen(
    prox.ChoiceMapDistribution,
    selection=genjax.AllSelection(),
)
def transition_proposal(key, prev_particle, target):
    (state,) = prev_particle.get_retval()
    observation = target.constraints["z", "observation"]
    config = target.args[1]
    trow = config.transition_tensor[state, :]
    orow = config.observation_tensor[:, observation]
    observation_weights = orow[jnp.arange(0, len(trow))]
    weights = trow + observation_weights
    key, _ = genjax.trace(("z", "latent"), genjax.Categorical)(key, (weights,))
    return (key,)


# Define an SMC propagator using a functional DSL.
@genjax.prox.smc.Propagator
def chain_propagator(key, state):
    def _inner(carry, x):
        key, state = carry
        new_args, new_constraints = x
        key, state = prox.smc.Extend(transition_proposal)(
            key, state, (new_args, new_constraints)
        )
        key, state = prox.smc.Resample(0.7, 50)(key, state, ())
        return (key, state), ()

    (key, state), () = jax.xla.scan(
        _inner,
        (key, state),
        length=50,
    )
    return key, state


#####
# Test
#####

key = jax.random.PRNGKey(314159)
config = genjax.DiscreteHMMConfiguration.new(10, 1, 1, 0.8, 0.8)

# Simulate ground truth.
key, tr = jax.jit(genjax.simulate(hidden_markov_model))(key, (2, config))
observation_sequence = tr["z", "observation"]

# Setup choice map observations.
chm_sequence = genjax.VectorChoiceMap.new(
    np.array([ind for ind in range(0, len(observation_sequence))]),
    genjax.ChoiceMap.new(
        {
            ("z", "observation"): np.array(
                observation_sequence, dtype=np.int32
            ),
        }
    ),
)


def choice_map_coercion(target, chm):
    args = target.args
    new_index = args[0]
    return IndexMask(new_index, chm)


# Build an initial posterior target.
initial_target = prox.Target(
    hidden_markov_model,
    choice_map_coercion,
    (0, config),
    IndexMask(0, jtu.tree_map(lambda v: v[0], chm_sequence.inner)),
)


# Compose an `smc.Init` step with propagation to create an
# `SMCAlgorithm`.
algorithm = genjax.prox.smc.Init(initial_proposal, 2).and_then(
    genjax.prox.smc.Extend(transition_proposal),
    (1, config),
    IndexMask(1, jtu.tree_map(lambda v: v[1], chm_sequence.inner)),
)

# Run the algorithm.
jitted = jax.jit(algorithm.simulate)
key, tr = jitted(key, (initial_target,))
console.print(tr)
