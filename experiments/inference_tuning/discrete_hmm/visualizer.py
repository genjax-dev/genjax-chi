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
import genjax.experimental.prox as prox
import numpy as np
from model_config import DiscreteHMMConfiguration, hidden_markov_model
from inference_config import (
    meta_initial_position,
    hmm_meta_next_target,
    transition_proposal,
)
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from typing import Sequence

plt.style.use("ggplot")

#####
# Visualizer
#####


def sequence_visualizer(
    ax,
    particle_label,
    ground_truth: Sequence,
    observations: Sequence,
    sequence: Sequence,
):
    if np.ndim(sequence) > 1:
        r = np.tile(np.arange(0, sequence.shape[1]), (sequence.shape[0], 1))
        ax.scatter(
            r,
            sequence,
            s=10,
            marker="s",
            color="darkred",
            alpha=0.1,
        )
    else:
        ax.scatter(
            range(0, len(sequence)),
            sequence,
            s=10,
            marker="s",
            color="darkred",
            alpha=0.05,
        )
    ax.scatter(
        range(0, len(observations)),
        ground_truth,
        s=20,
        marker="s",
        color="black",
        alpha=1.0,
    )
    ax.scatter(
        range(0, len(observations)),
        observations,
        s=10,
        marker="s",
        color="grey",
        alpha=1.0,
    )
    trans = mtransforms.ScaledTranslation(0 / 72, 0 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        particle_label,
        transform=ax.transAxes + trans,
        fontsize="medium",
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="1.0", edgecolor="none", pad=3.0),
    )


#####
# Inference
#####

key = jax.random.PRNGKey(314159)
num_steps = 30
config = DiscreteHMMConfiguration.new(30, 1, 1, 0.8, 0.2)

# Generate from model.
key, tr = jax.jit(genjax.simulate(hidden_markov_model))(
    key, (num_steps, config)
)
ground_truth = tr["z", "latent"]
observation_sequence = tr["z", "observation"]

# Observations.
chm_sequence = genjax.VectorChoiceMap.new(
    np.array([ind for ind in range(0, len(observation_sequence))]),
    genjax.ChoiceMap.new(
        {("z", "observation"): np.array(observation_sequence, dtype=np.int32)}
    ),
)

final_target = prox.Target(
    hidden_markov_model,
    None,
    (num_steps, config),
    chm_sequence,
)

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    sharex=True,
    figsize=(18, 18),
)
axes = axes.flatten()

for (ax, n_particles) in zip(
    axes,
    [1, 2, 5, 50, 100, 500, 1000, 1500, 2000],
    # [1, 2, 2, 2, 2, 2, 2, 2, 2],
):
    ax.set_ylim(-1, 31)
    ax.set_yticks([])
    ax.set_xticks([])
    custom_smc = genjax.CustomSMC(
        meta_initial_position,
        hmm_meta_next_target,
        transition_proposal,
        lambda _: num_steps,
        n_particles,
    )
    key, *sub_keys = jax.random.split(key, 100 + 1)
    sub_keys = jnp.array(sub_keys)
    _, tr = jax.vmap(jax.jit(custom_smc.simulate), in_axes=(0, None))(
        sub_keys, (final_target,)
    )
    (chm,) = tr.get_retval()
    sequence_visualizer(
        ax,
        n_particles,
        ground_truth,
        observation_sequence,
        chm["z", "latent"],
    )

plt.show()
