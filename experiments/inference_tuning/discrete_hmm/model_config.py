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
