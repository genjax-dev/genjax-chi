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
from model_config import hidden_markov_model
from inference_config import (
    meta_initial_position,
    hmm_meta_next_target,
    transition_proposal,
    prior_proposal,
)
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns
from typing import Sequence
from rich.progress import track

# Globals config.
sns.set()

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

#####
# Visualizer
#####


def sequence_visualizer(
    fig,
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
            s=7,
            marker="s",
            color="darkred",
            alpha=0.08,
            label="particles",
        )
    else:
        ax.scatter(
            range(0, len(sequence)),
            sequence,
            s=7,
            marker="s",
            color="darkred",
            alpha=0.08,
            label="particles",
        )
    ax.scatter(
        range(0, len(observations)),
        ground_truth,
        s=20,
        marker="s",
        color="black",
        alpha=1.0,
        label="ground_truth",
    )
    ax.scatter(
        range(0, len(observations)),
        observations,
        s=8,
        marker="s",
        color="grey",
        alpha=1.0,
        label="observation",
    )
    trans = mtransforms.ScaledTranslation(
        1 / 72,
        -3 / 72,
        fig.dpi_scale_trans,
    )
    ax.text(
        0.0,
        1.0,
        particle_label,
        transform=ax.transAxes + trans,
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="1.0", edgecolor="none", pad=3.0),
    )


#####
# Inference
#####

key = jax.random.PRNGKey(314159)
num_steps = 50
config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.15, 0.1)

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


def custom_smc_with_transition(n_particles):
    return genjax.CustomSMC(
        meta_initial_position,
        hmm_meta_next_target,
        transition_proposal,
        lambda _: num_steps,
        n_particles,
    )


def custom_smc_with_prior(n_particles):
    return genjax.CustomSMC(
        meta_initial_position,
        hmm_meta_next_target,
        prior_proposal,
        lambda _: num_steps,
        n_particles,
    )


def grid_plot(key, make_custom_smc):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        sharex=True,
        figsize=(18, 18),
        dpi=400,
    )
    axes = axes.flatten()

    for (ax, n_particles) in track(
        list(
            zip(
                axes,
                [1, 2, 5, 10, 20, 50, 100, 200, 500],
            )
        ),
        description="Creating grid plot...",
    ):
        ax.set_ylim(-1, config.linear_grid_dim)
        ax.set_xlim(-1, num_steps)
        ax.set_yticks([])
        ax.set_xticks([])
        custom_smc = make_custom_smc(n_particles)
        key, *sub_keys = jax.random.split(key, 300 + 1)
        sub_keys = jnp.array(sub_keys)
        _, tr = jax.vmap(jax.jit(custom_smc.simulate), in_axes=(0, None))(
            sub_keys, (final_target,)
        )
        (chm,) = tr.get_retval()
        sequence_visualizer(
            fig,
            ax,
            n_particles,
            ground_truth,
            observation_sequence,
            chm["z", "latent"],
        )

    labels_handles = {
        label: handle
        for ax in axes
        for handle, label in zip(*ax.get_legend_handles_labels())
    }

    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper right",
        # bbox_to_anchor=(0.5, 0),
        # bbox_transform=plt.gcf().transFigure,
    )
    return key, fig


# Run SMC with transition as proposal.
key, fig2 = grid_plot(key, custom_smc_with_transition)
fig2.suptitle("SMC (Locally optimal proposal)")
fig2.savefig("img/transition_proposal.png")

# Run SMC with prior as proposal.
key, fig1 = grid_plot(key, custom_smc_with_prior)
fig1.suptitle("SMC (Prior as proposal)")
fig1.savefig("img/prior_proposal.png")
