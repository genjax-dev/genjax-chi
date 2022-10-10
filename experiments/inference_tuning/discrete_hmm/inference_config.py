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
from model_config import initial_position
from model_config import kernel_step

import genjax
import genjax.experimental.prox as prox


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


@genjax.gen(
    prox.ChoiceMapDistribution,
    selection=genjax.AllSelection(),
)
def prior_proposal(key, state, new_target):
    v = state
    transition_tensor = new_target.args[1]
    trow = transition_tensor[v, :]
    key, _ = genjax.trace(("z", "latent"), genjax.Categorical)(key, (trow,))
    return (key,)


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


def meta_initial_position(final_state):
    config = final_state.args[1]
    return initial_position(config)
