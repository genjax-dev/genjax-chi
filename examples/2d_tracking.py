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
import numpy as np
import genjax
from genjax import Trace
from typing import Sequence
import timeit
import matplotlib.pyplot as plt

plt.style.use("ggplot")


# A 2D tracking example in GenJAX, with inference using propose-resample SMC.

transition_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
observation_matrix = np.array([[3.0, 0.0], [0.0, 3.0]])


# Note how we must specify a `max_length` for `UnfoldCombinator`
# here. This is required by JAX, so that it can statically reason
# about the static potential size of arrays.
@genjax.gen(genjax.Unfold, max_length=100)
def kernel(key, prev_latent):
    key, z = genjax.trace("latent", genjax.MvNormal)(
        key, (prev_latent, transition_matrix)
    )
    key, x = genjax.trace("obs", genjax.MvNormal)(key, (z, observation_matrix))
    return key, z


@genjax.gen
def model(key, length):
    key, initial_latent = genjax.trace("initial", genjax.MvNormal)(
        key, (np.array([0.0, 0.0]), observation_matrix)
    )
    key, z = genjax.trace("z", kernel)(
        key,
        (length, initial_latent),
    )
    return key, z


key = jax.random.PRNGKey(314159)
observation_sequence = np.array(
    [
        [i, i] if i % 2 == 0 else [i, -i] if i < 50 else [100 - 3 * i, 3 * i]
        for i in range(0, 100)
    ]
)

trace_type = genjax.get_trace_type(model)(key, (100,))
print(trace_type)


#####
# Inference
#####

# Here's a simple trace visualizer which plots the latent
# sequence from a trace against the observation sequence.
def trace_visualizer(observation_sequence: Sequence, tr: Trace):
    fig, ax = plt.subplots()
    value = tr[("z", "latent")]
    latent_x = value[:, :, 0]
    latent_y = value[:, :, 1]
    obs_x = observation_sequence[:, 0]
    obs_y = observation_sequence[:, 1]
    plt.scatter(latent_x, latent_y, marker=".")
    plt.scatter(obs_x, obs_y, marker="x")
    plt.title("Observation sequence vs. particle traces")
    plt.xlabel("x")
    plt.ylabel("y")
    fig.set_size_inches(10, 8)
    plt.show()


#####
# Our inference program will utilize a custom initial and transition
# proposal.
#####


# The `Partial` combinator closes over arguments -- allowing
# JAX to implement constant propagation.
#
# `Partial` always closes from last to first argument.
#
# Thus, here we're indicating that `obs_chm` must be static underneath
# any JAX transformation.
#
# Note that JAX will be very upset if you don't pass in a constant
# argument (as you said you would).
#
# Then, a closure will capture a tracer, which is illegal.
@genjax.gen(genjax.Partial)
def initial_proposal(key, obs_chm):
    v = obs_chm["z", "obs"]
    key, initial = genjax.trace("initial", genjax.MvNormal)(
        key, (v, observation_matrix)
    )
    key, first_latent = genjax.trace(("z", "latent"), genjax.MvNormal)(
        key, (initial, observation_matrix)
    )
    return (key,)


@genjax.gen(genjax.Partial)
def transition_proposal(key, prev_tr, obs_chm):
    v = obs_chm["z", "obs"]
    key, first_latent = genjax.trace(("z", "latent"), genjax.MvNormal)(
        key, (v, transition_matrix)
    )
    return (key,)


# Here's a convenient way to specify a sequence of observations
# for an algorithm like SMC -- use a `VectorChoiceMap` to store
# the observations, and create a static index mask
# which will isolate each individual contributed observation
# (over the time index)
chm_sequence = genjax.VectorChoiceMap(
    genjax.ChoiceMap.new({("z", "obs"): np.array(observation_sequence)})
)

# SMC allows a progression of different target measures --
# here, we parametrize that progression using a sequence of different
# arguments to the model.
model_arg_sequence = [(ind,) for ind in range(1, len(observation_sequence) + 1)]


# Run inference.
jitted = jax.jit(
    genjax.proposal_sequential_monte_carlo(
        model, initial_proposal, transition_proposal, 100
    ),
    static_argnums=1,
)

key, (tr, lmle) = jitted(
    key, chm_sequence, model_arg_sequence, [() for _ in model_arg_sequence]
)

trace_visualizer(observation_sequence, tr)
