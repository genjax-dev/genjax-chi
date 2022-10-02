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
import matplotlib.ticker as ticker
import seaborn as sns
import ptitprince as pt
from rich.progress import track


# Global setup.
key = jax.random.PRNGKey(314159)
num_steps = 50

sns.set()

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
# Custom configs
#####


def make_config(t):
    transition_delta = t[0]
    observation_delta = t[1]
    config = genjax.DiscreteHMMConfiguration.new(
        50, 2, 1, transition_delta, observation_delta
    )
    return config


#####
# IEE
#####


def run_iee_experiment(key):
    inf_selection = genjax.Selection([("z", "latent")])
    iee_jitted = jax.jit(
        genjax.iee(
            hidden_markov_model,
            exact_hmm_posterior,
            inf_selection,
            4000,
            1,
        )
    )

    def sdos_visualizer(axes, df):
        dx, dy = "entropy", "est"
        pt.RainCloud(
            ax=axes,
            x=dx,
            y=dy,
            data=df,
            bw=0.2,
            scale="width",
            width_viol=1.0,
            box_showfliers=False,
        )

    def sdos_for_nparticles(key, config, custom_smc):
        inf_selection = genjax.Selection([("z", "latent")])
        key, ratio, (fwd, bwd) = genjax.sdos(
            hidden_markov_model,
            custom_smc,
            inf_selection,
            1,
            400,
        )(key, (num_steps, config))
        return key, (jnp.mean(ratio), jnp.sqrt(jnp.var(ratio))), ratio

    def sdos_plot(key, axes, custom_smc, config_sequence):
        d = {"entropy": [], "est": []}
        df = pd.DataFrame(d)
        custom_configs = list(map(make_config, config_sequence))

        for custom_config in track(
            custom_configs, description="SDOS by entropy"
        ):
            ratio = np.array([], dtype=np.float32)

            def _lambda(key, custom_config, custom_smc):
                key, *sub_keys = jax.random.split(key, 100 + 1)
                sub_keys = jnp.array(sub_keys)
                _, (_, _), r = jax.vmap(
                    sdos_for_nparticles, in_axes=(0, None, None)
                )(sub_keys, custom_config, custom_smc)
                return key, r

            jitted = jax.jit(_lambda)
            for _ in range(0, 4):
                key, r = jitted(key, custom_config, custom_smc)
                ratio = np.append(ratio, np.array(r))
            key, entropy, _ = iee_jitted(key, (num_steps, custom_config))
            console.print(custom_config)
            console.print(entropy)
            console.print(jnp.mean(ratio))
            entropy = (entropy[0] + entropy[1]) / 2
            dx = np.repeat(entropy, len(ratio))
            d = {"entropy": dx, "est": ratio}
            new = pd.DataFrame(data=d)
            df = pd.concat([df, new], ignore_index=True)

        df = df.astype({"entropy": int, "est": float})
        sdos_visualizer(axes, df)

        return key

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(18, 12), dpi=400)

    config_sequence = [
        (0.2, 0.1),
        (0.2, 0.15),
        (0.2, 0.2),
        (0.2, 0.25),
        (0.2, 0.3),
        (0.2, 0.35),
        (0.2, 0.4),
    ]

    # Run SMC with prior as proposal.
    _ = sdos_plot(key, ax1, custom_smc_with_prior(100), config_sequence)
    ax1.set_xlabel("Entropy estimate")
    ax1.set_ylabel("Estimator")
    ax1.set_title("SMC (100 particles, prior as proposal)")

    # Run SMC with transition as proposal.
    key = sdos_plot(key, ax2, custom_smc_with_transition(100), config_sequence)
    ax2.set_xlabel("Entropy estimate")
    ax2.set_ylabel("")
    ax2.set_title("SMC (100 particles, locally optimal proposal)")

    fig.suptitle(
        "Symmetric Divergence over Datasets (SDOS) vs. entropy\n(Varying over model observation noise)"
    )

    plt.savefig("img/entropy_sdos_obs.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(18, 12), dpi=400)

    config_sequence = [
        (0.05, 0.2),
        (0.1, 0.2),
        (0.15, 0.2),
        (0.2, 0.2),
        (0.25, 0.2),
        (0.3, 0.2),
        (0.35, 0.2),
        (0.4, 0.2),
        (0.45, 0.2),
        (0.5, 0.2),
    ]

    # Run SMC with prior as proposal.
    _ = sdos_plot(key, ax1, custom_smc_with_prior(100), config_sequence)
    ax1.set_xlabel("Entropy estimate")
    ax1.set_ylabel("Estimator")
    ax1.set_title("SMC (100 particles, prior as proposal)")

    # Run SMC with transition as proposal.
    key = sdos_plot(key, ax2, custom_smc_with_transition(100), config_sequence)
    ax2.set_xlabel("Entropy estimate")
    ax2.set_ylabel("")
    ax2.set_title("SMC (100 particles, locally optimal proposal)")

    fig.suptitle(
        "Symmetric Divergence over Datasets (SDOS) vs. entropy\n(Varying over model transition noise)"
    )

    plt.savefig("img/entropy_sdos_trans.png")

    return key


#####
# Experiments
#####

key = run_iee_experiment(key)
