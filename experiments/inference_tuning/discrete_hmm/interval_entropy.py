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
import pandas as pd
import genjax
from model_config import hidden_markov_model
from inference_config import (
    meta_initial_position,
    hmm_meta_next_target,
    transition_proposal,
    prior_proposal,
)
import genjax.experimental.prox as prox
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
from rich.progress import track


# Global setup.
key = jax.random.PRNGKey(314159)
num_steps = 50

sns.set()

SMALL_SIZE = 36
MEDIUM_SIZE = 40
BIGGER_SIZE = 44

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["text.usetex"] = True  # Use LaTeX.

# Set pretty printing + tracebacks.
console = genjax.go_pretty()

#####
# Exact inference
#####


@genjax.gen(
    prox.ChoiceMapDistribution,
    selection=genjax.AllSelection(),
)
def exact_hmm_posterior(key, target):
    config = target.args[1]
    observations = target.constraints["z", "observation"]
    key, v = genjax.trace(("z", "latent"), genjax.DiscreteHMM)(
        key, (config, observations)
    )
    return (key,)


#####
# SMC variants
#####


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


#####
# Raincloud plot
#####


def raincloud_plot(axes, df, dx, dy):
    pt.RainCloud(
        x=dx,
        y=dy,
        data=df,
        bw=0.2,
        scale="area",
        width_viol=1.0,
        box_showfliers=False,
        ax=axes,
    )


#####
# eevi
#####

inf_selection = genjax.Selection([("z", "latent")])


def run_eevi(key, inference, config):
    key, (lower, upper), _ = genjax.iee(
        hidden_markov_model,
        inference,
        inf_selection,
        1000,
        1,
    )(key, (num_steps, config))
    entropy = (lower + upper) / 2
    return key, entropy


jitted = jax.jit(run_eevi)


def run_eevi_experiment(key, config):
    def eevi_plot(key, axes, make_custom_smc):
        d = {"particles": [], "est": []}
        df = pd.DataFrame(d)
        key, exact_inf_entropy = jitted(key, exact_hmm_posterior, config)
        vmap_jitted = jax.jit(jax.vmap(run_eevi, in_axes=(0, None, None)))
        for n_particles in track(
            [1, 2, 5, 10, 20, 50, 100, 200], "Entropy by num. particles"
        ):
            custom_smc = make_custom_smc(n_particles)
            for _ in range(0, 10):
                key, *sub_keys = jax.random.split(key, 10)
                sub_keys = jnp.array(sub_keys)
                _, means = vmap_jitted(sub_keys, custom_smc, config)
                d = {
                    "particles": np.repeat(n_particles, len(means)),
                    "est": means,
                }
                new = pd.DataFrame(data=d)
                df = pd.concat([df, new], ignore_index=True)
                df = df.astype({"particles": int, "est": float})

        raincloud_plot(axes, df, "particles", "est")
        axes.axhline(y=exact_inf_entropy, linewidth=5, label="Exact inference")
        axes.set_xlabel(r"$\#$ of particles")
        axes.set_ylabel("$p(x)$ entropy estimate")
        return key

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(36, 18),
        dpi=400,
    )
    key = eevi_plot(key, ax1, custom_smc_with_prior)
    key = eevi_plot(key, ax2, custom_smc_with_transition)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_ylabel("")
    fig.suptitle(
        "EEVI (model-average posterior entropy) convergence"
        "\n(SMC, prior as proposal) vs. (SMC, locally optimal proposal)"
    )
    labels_handles = {
        label: handle for handle, label in zip(*ax2.get_legend_handles_labels())
    }

    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper right",
    )
    plt.savefig("img/eevi_smc_convergence.png")

    return key


#####
# Experiments
#####

config = genjax.DiscreteHMMConfiguration.new(50, 1, 1, 0.1, 0.1)
key = run_eevi_experiment(key, config)
