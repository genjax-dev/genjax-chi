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
import genjax
from genjax import Trace
from typing import Sequence

# A 2D tracking example in GenJAX, with inference using sequential Monte Carlo.

transition_matrix = jnp.array([[3.0, 0.0], [0.0, 3.0]])
observation_matrix = jnp.array([[0.3, 0.0], [0.0, 0.3]])


@genjax.gen(genjax.UnfoldCombinator, length=8)
def kernel(key, prev_latent):
    key, z = genjax.trace("latent", genjax.MvNormal)(
        key, (prev_latent, transition_matrix)
    )
    key, x = genjax.trace("obs", genjax.MvNormal)(key, (z, observation_matrix))
    return key, z


@genjax.gen
def model(key):
    key, initial_latent = genjax.trace("initial", genjax.Uniform, shape=(2,))(
        key,
        (-3.0, 3.0),
    )
    key, z = genjax.trace("z", kernel)(key, (initial_latent,))
    return key, z


observation_sequence = jnp.array(
    [
        [-2.0, -2.0],
        [2.0, 2.0],
        [4.0, 4.0],
        [6.0, 6.0],
        [8.0, 8.0],
        [10.0, 10.0],
        [12.0, 12.0],
        [14.0, 14.0],
    ]
)


#####
# Visualizer
#####

import matplotlib.pyplot as plt

plt.style.use("ggplot")


def trace_visualizer(observation_sequence: Sequence, tr: Trace):
    fig, ax = plt.subplots()
    value = tr[("z", "latent")]
    latent_x = value[:, :, 0]
    latent_y = value[:, :, 1]
    obs_x = observation_sequence[:, 0]
    obs_y = observation_sequence[:, 1]
    plt.scatter(latent_x, latent_y, marker=".")
    plt.scatter(obs_x, obs_y, marker=".")
    plt.title("Observation sequence vs. trace")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.set_size_inches(10, 8)
    plt.show()


# Here's an implementation of jittable importance resampling, using
# the prior as a proposal.
def importance_resampling(model, key, args, obs, n_particles):
    key, *subkeys = jax.random.split(key, n_particles + 1)
    subkeys = jnp.array(subkeys)
    _, (lws, trs) = jax.vmap(genjax.importance(model), in_axes=(0, None, None))(
        subkeys, obs, args
    )
    log_total_weight = jax.scipy.special.logsumexp(lws)
    log_normalized_weights = lws - log_total_weight
    ind = jax.random.categorical(key, log_normalized_weights)
    tr = jax.tree_util.tree_map(lambda v: v[ind], trs)
    lw = lws[ind]
    return lw, tr


chm = genjax.ChoiceMap(
    {("z",): genjax.VectorChoiceMap({("obs",): observation_sequence}, 8)},
)
key = jax.random.PRNGKey(314159)
key, *subkeys = jax.random.split(key, 1000 + 1)
subkeys = jnp.array(subkeys)
ws, trs = jax.vmap(
    jax.jit(importance_resampling, static_argnums=4),
    in_axes=(None, 0, None, None, None),
)(model, subkeys, (), chm, 10000)

trace_visualizer(observation_sequence, trs)
