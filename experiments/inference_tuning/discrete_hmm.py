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
import numpy as np
from scipy.linalg import circulant
from dataclasses import dataclass
import genjax
from genjax.core.pytree import Pytree
from typing import Union, Sequence
import matplotlib.pyplot as plt

plt.style.use("ggplot")

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
class DiscreteHMMConfiguration(Pytree):
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
            sigma_trans,
            1 / sigma_trans,
        )

        observation_tensor = scaled_circulant(
            linear_grid_dim,
            adjacency_distance_obs,
            sigma_obs,
            1 / sigma_obs,
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
# Visualizer
#####


def sequence_visualizer(sequence: Sequence):
    fig, _ = plt.subplots()
    plt.scatter(range(0, len(sequence)), sequence)
    plt.title("State sequence vs. time")
    plt.xlabel("Time")
    plt.ylabel("State")
    fig.set_size_inches(10, 8)
    plt.show()


#####
# Model
#####

# Below, `genjax.Categorical` expects logit tensors,
# not normalized probability tensors, so `trow`
# and `orow` are not normalized.
@genjax.gen(genjax.Unfold, max_length=100)
def kernel(key, prev, transition_tensor, observation_tensor):
    trow = transition_tensor[prev, :]
    key, latent = genjax.trace("latent", genjax.Categorical)(key, (trow,))
    orow = observation_tensor[latent, :]
    key, observation = genjax.trace("observation", genjax.Categorical)(
        key, (orow,)
    )
    return key, latent


def initial_position(config: DiscreteHMMConfiguration):
    return jnp.array(int(config.linear_grid_dim / 2))


@genjax.gen
def hidden_markov_model(key, T, config):
    transition_tensor = config.transition_tensor
    observation_tensor = config.observation_tensor
    z0 = initial_position(config)
    key, z = genjax.trace("z", kernel)(
        key, (T, z0, transition_tensor, observation_tensor)
    )
    return key, z


key = jax.random.PRNGKey(314159)
num_steps = 100
config = DiscreteHMMConfiguration.new(10, 1, 1, 0.1, 0.1)
key, tr = jax.jit(genjax.simulate(hidden_markov_model))(
    key, (num_steps, config)
)
(chm,) = tr.get_retval()
sequence = chm[("z", "latent")]
sequence_visualizer(sequence)

#####
# Inference
#####


def hmm_meta_next_target(state, constraints, final_target):
    pass


@genjax.gen
def hmm_meta_proposal(state, new_target, final_target):
    pass


custom_smc = genjax.CustomSMC(
    initial_position,
    hmm_meta_next_target,
    hmm_meta_proposal,
    lambda _: num_steps,
    50,
)
