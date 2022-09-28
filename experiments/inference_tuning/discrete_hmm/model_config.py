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
import numpy as np
from scipy.linalg import circulant
from dataclasses import dataclass
import genjax
from typing import Union

Int = int
Float32 = Union[np.float32, jnp.float32]
FloatTensor = Union[jnp.ndarray, np.ndarray]


def scaled_circulant(N, k, epsilon, delta):
    source = [
        epsilon ** abs(index)
        if index <= k
        else epsilon ** abs(index - N)
        if index - N >= -k
        else -delta
        for index in range(0, N)
    ]
    return circulant(source)


@dataclass
class DiscreteHMMConfiguration(genjax.Pytree):
    linear_grid_dim: Int
    adjacency_distance_trans: Float32
    adjacency_distance_obs: Int
    sigma_trans: Float32
    sigma_obs: Float32
    transition_tensor: FloatTensor
    observation_tensor: FloatTensor

    def flatten(self):
        return (self.transition_tensor, self.observation_tensor,), (
            self.linear_grid_dim,
            self.adjacency_distance_trans,
            self.adjacency_distance_obs,
            self.sigma_trans,
            self.sigma_obs,
        )

    @classmethod
    def new(
        cls,
        linear_grid_dim: Int,
        adjacency_distance_trans: Float32,
        adjacency_distance_obs: Float32,
        sigma_trans: Float32,
        sigma_obs: Float32,
    ):
        transition_tensor = scaled_circulant(
            linear_grid_dim,
            adjacency_distance_trans,
            sigma_trans if sigma_trans > 0.0 else -np.inf,
            1 / sigma_trans if sigma_trans > 0.0 else -np.inf,
        )

        observation_tensor = scaled_circulant(
            linear_grid_dim,
            adjacency_distance_obs,
            sigma_obs if sigma_obs > 0.0 else -np.inf,
            1 / sigma_obs if sigma_obs > 0.0 else np.inf,
        )
        return DiscreteHMMConfiguration(
            linear_grid_dim,
            adjacency_distance_trans,
            adjacency_distance_obs,
            sigma_trans,
            sigma_obs,
            transition_tensor,
            observation_tensor,
        )


#####
# Model
#####


@genjax.gen
def kernel_step(key, prev, config):
    transition_tensor = config.transition_tensor
    observation_tensor = config.observation_tensor
    trow = transition_tensor[prev, :]
    key, latent = genjax.trace("latent", genjax.Categorical)(key, (trow,))
    orow = observation_tensor[latent, :]
    key, observation = genjax.trace("observation", genjax.Categorical)(
        key, (orow,)
    )
    return key, latent


kernel = genjax.Unfold(kernel_step, max_length=2)


def initial_position(config: DiscreteHMMConfiguration):
    return jnp.array(int(config.linear_grid_dim / 2))


@genjax.gen
def hidden_markov_model(key, T, config):
    z0 = initial_position(config)
    key, z = genjax.trace("z", kernel)(key, (T, z0, config))
    return key, z
