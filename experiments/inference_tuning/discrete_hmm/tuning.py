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

from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
from inference_config import hmm_meta_next_target
from inference_config import meta_initial_position
from inference_config import prior_proposal
from inference_config import transition_proposal
from model_config import hidden_markov_model
from rich.progress import track

import genjax
import genjax.experimental.prox as prox


# Global setup.
key = jax.random.PRNGKey(314159)
num_steps = 50
config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.15, 0.1)

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
# SDOS
#####


def run_sdos_experiment(key):
    def sdos_for_nparticles(key, custom_smc):
        inf_selection = genjax.Selection([("z", "latent")])
        key, ratio, (fwd, bwd) = genjax.sdos(
            hidden_markov_model,
            custom_smc,
            inf_selection,
            1,
            100,
        )(key, (num_steps, config))
        return key, (jnp.mean(ratio), jnp.sqrt(jnp.var(ratio))), ratio

    def sdos_plot(key, axes, make_custom_smc):
        d = {"particles": [], "est": []}
        df = pd.DataFrame(d)
        custom_smcs = list(
            map(make_custom_smc, [1, 2, 5, 10, 20, 50, 100, 200, 500])
        )

        for custom_smc in track(
            custom_smcs, description="SDOS by # particles"
        ):
            ratio = np.array([], dtype=np.float32)

            def _lambda(key, custom_smc):
                key, *sub_keys = jax.random.split(key, 10 + 1)
                sub_keys = jnp.array(sub_keys)
                _, (_, _), r = jax.vmap(
                    sdos_for_nparticles, in_axes=(0, None)
                )(sub_keys, custom_smc)
                return key, r

            jitted = jax.jit(_lambda)
            for _ in range(0, 50):
                key, r = jitted(key, custom_smc)
                ratio = np.append(ratio, np.array(r))
            dx = np.repeat(custom_smc.num_particles, len(ratio))
            d = {"particles": dx, "est": ratio}
            new = pd.DataFrame(data=d)
            df = pd.concat([df, new], ignore_index=True)

        df = df.astype({"particles": int, "est": float})
        raincloud_plot(axes, df, "particles", "est")

        return key

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, figsize=(18, 12), dpi=400
    )

    # Run SMC with prior as proposal.
    key = sdos_plot(key, ax1, custom_smc_with_prior)
    ax1.set_xlabel("Num. of particles")
    ax1.set_ylabel("Estimator")
    ax1.set_title("SMC (Prior as proposal)")

    # Run SMC with transition as proposal.
    key = sdos_plot(key, ax2, custom_smc_with_transition)
    ax2.set_xlabel("Num. of particles")
    ax2.set_ylabel("")
    ax2.set_title("SMC (Locally optimal proposal)")

    fig.suptitle("Symmetric Divergence over Datasets (SDOS)")

    plt.savefig("img/sdos.png")
    return key


#####
# AIDE
#####


def run_aide_experiment(key):
    def sample_and_score(key, config):
        selection = genjax.Selection([("z", "observation")])
        key, tr = genjax.simulate(hidden_markov_model)(
            key, (num_steps, config)
        )
        chm = tr.get_choices().strip()
        observations = chm["z", "observation"]
        logprob = genjax.DiscreteHMM.data_logpdf(config, observations)
        chm, _ = selection.filter(chm)
        return key, (logprob, chm)

    def aide_threshold_plot(
        key, make_custom_smc, probability_failure, kl_threshold
    ):
        fig, axes = plt.subplots(figsize=(24, 18), dpi=400)
        d = {"particles": [], "est": []}
        df = pd.DataFrame(d)

        def _lambda(key, custom_smc):
            key, (exact_score, chm) = sample_and_score(key, config)
            final_target = prox.Target(
                hidden_markov_model,
                None,
                (num_steps, config),
                chm,
            )
            observations = chm["z", "observation"]
            key, *sub_keys = jax.random.split(key, 100 + 1)
            sub_keys = jnp.array(sub_keys)
            _, est, (_, _) = jax.vmap(
                genjax.aide(exact_hmm_posterior, custom_smc, 1, 50),
                in_axes=(0, None, None),
            )(sub_keys, (final_target,), (final_target,))
            return key, (exact_score, est)

        jitted = jax.jit(_lambda)
        num_particles = 10
        empirical_prob_fail = 1.0
        while empirical_prob_fail > probability_failure:
            means = np.array([], dtype=np.float32)
            exact_score = np.array([], dtype=np.float32)
            custom_smc = make_custom_smc(num_particles)
            for _ in track(range(0, 100), description="Tuning w/ AIDE"):
                key, (es, r) = jitted(key, custom_smc)
                means = np.append(means, jnp.mean(r))
                exact_score = np.append(exact_score, np.array(es))
            particles = np.repeat(num_particles, len(means))
            d = {
                "particles": particles,
                "est": means,
            }
            new = pd.DataFrame(data=d)
            df = pd.concat([df, new], ignore_index=True)
            df = df.astype({"particles": int, "est": float})
            total = jnp.sum(jax.vmap(lambda v: v < kl_threshold)(means))
            console.print(means)
            console.print((len(means), total))
            empirical_prob_fail = 1 - (total / len(means))
            num_particles += 10

        raincloud_plot(axes, df, "particles", "est")
        axes.axhline(y=kl_threshold, linewidth=5)
        return key, fig

    # AIDE threshold plot with prior as proposal.
    key, fig = aide_threshold_plot(key, custom_smc_with_prior, 0.3, 1.0)
    fig.suptitle(
        "AIDE tuning (SMC, prior as proposal) against exact inference"
        "\n(p_failure=0.3, KL threshold = 1.0, 100 samples from model)"
    )
    plt.xlabel("Num. of particles")
    plt.ylabel("Estimator")
    plt.savefig("img/tuning_100_aide_prior.png")

    # AIDE threshold plot with prior as proposal.
    key, fig = aide_threshold_plot(key, custom_smc_with_transition, 0.3, 1.0)
    fig.suptitle(
        "AIDE tuning (SMC, locally optimal proposal) against exact inference"
        "\n(p_failure=0.3, KL threshold = 1.0, 100 samples from model)"
    )
    plt.xlabel("Num. of particles")
    plt.ylabel("Estimator")
    plt.savefig("img/tuning_100_aide_transition.png")

    return key


#####
# HMM quad plot
#####


def run_hmm_quad_plot(key, gold_standard, config):
    def sample_and_score(key, config):
        selection = genjax.Selection([("z", "observation")])
        key, tr = genjax.simulate(hidden_markov_model)(
            key, (num_steps, config)
        )
        chm = tr.get_choices().strip()
        observations = chm["z", "observation"]
        logprob = genjax.DiscreteHMM.data_logpdf(config, observations)
        filtered, _ = selection.filter(chm)
        return key, (logprob, filtered), chm

    def sdos_for_nparticles(key, custom_smc):
        inf_selection = genjax.Selection([("z", "latent")])
        key, ratio, (fwd, bwd) = genjax.sdos(
            hidden_markov_model,
            custom_smc,
            inf_selection,
            1,
            100,
        )(key, (num_steps, config))
        return key, (jnp.mean(ratio), jnp.sqrt(jnp.var(ratio))), ratio

    def sdos_plot(key, axes, make_custom_smc):
        d = {"particles": [], "est": []}
        df = pd.DataFrame(d)
        custom_smcs = list(
            map(make_custom_smc, [1, 2, 5, 10, 20, 50, 100, 200, 500])
        )

        for custom_smc in track(
            custom_smcs, description="SDOS by # particles"
        ):
            ratio = np.array([], dtype=np.float32)

            def _lambda(key, custom_smc):
                key, *sub_keys = jax.random.split(key, 10 + 1)
                sub_keys = jnp.array(sub_keys)
                _, (_, _), r = jax.vmap(
                    sdos_for_nparticles, in_axes=(0, None)
                )(sub_keys, custom_smc)
                return key, r

            jitted = jax.jit(_lambda)
            for _ in range(0, 50):
                key, r = jitted(key, custom_smc)
                ratio = np.append(ratio, np.array(r))
            dx = np.repeat(custom_smc.num_particles, len(ratio))
            d = {"particles": dx, "est": ratio}
            new = pd.DataFrame(data=d)
            df = pd.concat([df, new], ignore_index=True)

        df = df.astype({"particles": int, "est": float})
        raincloud_plot(axes, df, "particles", "est")
        return key

    def aide_plot(key, axes, gold_standard, custom_smc):
        axes.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        d = {"logpdf": [], "est": []}
        df = pd.DataFrame(d)
        ratio = np.array([], dtype=np.float32)
        score = np.array([], dtype=np.float32)
        chms = []
        smc_chms = []

        def _lambda(key, gold_standard, custom_smc):
            key, (score, filtered), chm = sample_and_score(key, config)
            final_target = prox.Target(
                hidden_markov_model,
                None,
                (num_steps, config),
                filtered,
            )
            observations = chm["z", "observation"]
            key, *sub_keys = jax.random.split(key, 100 + 1)
            sub_keys = jnp.array(sub_keys)
            _, est, (_, _) = jax.vmap(
                genjax.aide(gold_standard, custom_smc, 20, 200),
                in_axes=(0, None, None),
            )(sub_keys, (final_target,), (final_target,))
            score = jnp.repeat(score, len(est))
            key, *sub_keys = jax.random.split(key, 10 + 1)
            sub_keys = jnp.array(sub_keys)
            _, tr = jax.vmap(custom_smc.simulate, in_axes=(0, None))(
                sub_keys, (final_target,)
            )
            (smc_chm,) = tr.get_retval()
            return key, (score, est), (chm, smc_chm)

        jitted = jax.jit(_lambda)
        for _ in track(range(0, 3), description="Sample + AIDE"):
            key, (es, r), (chm, smc_chm) = jitted(
                key, gold_standard, custom_smc
            )
            ratio = np.append(ratio, np.array(r))
            score = np.append(score, np.array(es))
            chms.append(chm)
            smc_chms.append(smc_chm)

        d = {
            "logpdf": np.around(score, decimals=2),
            "est": ratio,
        }
        new = pd.DataFrame(data=d)
        df = pd.concat([df, new], ignore_index=True)

        raincloud_plot(axes, df, "logpdf", "est")

        return key, (chms, smc_chms)

    def sequence_plot(
        fig,
        ax,
        particle_label,
        ground_truth: Sequence,
        observations: Sequence,
    ):
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
            10 / 72,
            -10 / 72,
            fig.dpi_scale_trans,
        )
        ax.text(
            0.0,
            1.0,
            particle_label,
            transform=ax.transAxes + trans,
            verticalalignment="top",
            fontfamily="serif",
            fontsize="medium",
            bbox=dict(facecolor="1.0", edgecolor="none", pad=3.0),
        )

    fig = plt.figure(figsize=(48, 20), dpi=600)
    grid = plt.GridSpec(2, 4, wspace=0.35, hspace=0.4)
    grid_sub = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=grid[:, 0:2]
    )

    # Logpdf histogram.
    key, *sub_keys = jax.random.split(key, 2000 + 1)
    sub_keys = jnp.array(sub_keys)
    _, (logprob, chm), _ = jax.jit(
        jax.vmap(sample_and_score, in_axes=(0, None))
    )(sub_keys, config)
    ax1 = plt.subplot(grid_sub[1:, :])
    ax1.hist(logprob)
    ax1.set_xlabel(r"$\log p(x)$")
    ax1.set_ylabel("Counts")
    ax1.set_title(r"$\log p(x)$, repeated model samples")

    # Prior SDOS.
    ax2 = plt.subplot(grid[0, 2])
    key = sdos_plot(key, ax2, custom_smc_with_prior)
    ax2.set_xlabel("Num. of particles")
    ax2.set_ylabel("Estimator")
    ax2.set_title("SDOS (SMC, prior as proposal)")

    # Proposal SDOS.
    ax3 = plt.subplot(grid[0, 3])
    ax3.sharey(ax2)
    plt.setp(ax3.get_yticklabels(), visible=False)
    _ = sdos_plot(key, ax3, custom_smc_with_transition)
    ax3.set_xlabel("Num. of particles")
    ax3.set_ylabel("")
    ax3.set_title("SDOS (SMC, locally optimal proposal)")

    # Prior AIDE.
    ax4 = plt.subplot(grid[1, 2])
    _, _ = aide_plot(key, ax4, gold_standard, custom_smc_with_prior(100))
    ax4.set_xlabel(r"$\log p(x)$")
    ax4.set_ylabel("Estimator")
    ax4.set_title("AIDE (SMC, 100 particles, prior as proposal)")

    # Proposal AIDE.
    ax5 = plt.subplot(grid[1, 3])
    ax5.sharey(ax4)
    plt.setp(ax5.get_yticklabels(), visible=False)
    _, (chms, smc_chms) = aide_plot(
        key, ax5, gold_standard, custom_smc_with_transition(100)
    )
    ax5.set_xlabel(r"$\log p(x)$")
    ax5.set_ylabel("")
    ax5.set_title("AIDE (SMC, 100 particles, locally optimal proposal)")

    # Plot trajectories.
    chm1, chm2, chm3 = chms
    smc_chm1, smc_chm2, smc_chm3 = smc_chms
    ax6 = plt.subplot(grid_sub[0, 0])
    ax7 = plt.subplot(grid_sub[0, 1])
    ax8 = plt.subplot(grid_sub[0, 2])

    gt = chm1["z", "latent"]
    gt_obs = chm1["z", "observation"]
    logpdf_data = genjax.DiscreteHMM.data_logpdf(config, gt_obs).astype(int)
    sequence_plot(
        fig,
        ax6,
        r"$\log p(x) \sim$ " f"{logpdf_data}",
        gt,
        gt_obs,
    )
    ax6.set_ylim(-1, config.linear_grid_dim)
    ax6.set_xlim(-1, num_steps)
    ax6.set_yticks([])
    ax6.set_xticks([])

    gt = chm2["z", "latent"]
    gt_obs = chm2["z", "observation"]
    logpdf_data = genjax.DiscreteHMM.data_logpdf(config, gt_obs).astype(int)
    sequence_plot(
        fig,
        ax7,
        f"{logpdf_data}",
        gt,
        gt_obs,
    )
    ax7.set_ylim(-1, config.linear_grid_dim)
    ax7.set_xlim(-1, num_steps)
    ax7.set_yticks([])
    ax7.set_xticks([])

    gt = chm3["z", "latent"]
    gt_obs = chm3["z", "observation"]
    logpdf_data = genjax.DiscreteHMM.data_logpdf(config, gt_obs).astype(int)
    sequence_plot(
        fig,
        ax8,
        f"{logpdf_data}",
        gt,
        gt_obs,
    )
    ax8.set_ylim(-1, config.linear_grid_dim)
    ax8.set_xlim(-1, num_steps)
    ax8.set_yticks([])
    ax8.set_xticks([])

    return key, fig


#####
# Run experiments
#####

# key = run_sdos_experiment(key)
key = run_aide_experiment(key)

# Exact inference as gold standard.
# gold_standard = exact_hmm_posterior
# config = genjax.DiscreteHMMConfiguration.new(50, 1, 1, 0.8, 0.8)
# key, fig = run_hmm_quad_plot(key, gold_standard, config)
# fig.suptitle(
#    "Discrete HMM ($\#$ latent states = 50, $\#$ obs. states = 50, T = 50), high transition / obs entropy"
# )
# plt.savefig("img/exact_high_entropy_hmm_quad_plot.png")
#
# config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.1, 0.9)
# key, fig = run_hmm_quad_plot(key, gold_standard, config)
# fig.suptitle(
#    "Discrete HMM ($\#$ latent states = 50, $\#$ obs. states = 50, T = 50), low transition / high obs entropy"
# )
# plt.savefig("img/exact_mixed_entropy_hmm_quad_plot.png")
#
# config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.1, 0.1)
# key, fig = run_hmm_quad_plot(key, gold_standard, config)
# fig.suptitle(
#    "Discrete HMM ($\#$ latent states = 50, $\#$ obs. states = 50, T = 50), low transition / obs entropy"
# )
# plt.savefig("img/exact_low_entropy_hmm_quad_plot.png")
#
## SMC as gold standard.
# gold_standard = custom_smc_with_transition(200)
# config = genjax.DiscreteHMMConfiguration.new(50, 1, 1, 0.8, 0.8)
# key, fig = run_hmm_quad_plot(key, gold_standard, config)
# fig.suptitle(
#    "Discrete HMM ($\#$ latent states = 50, $\#$ obs. states = 50, T = 50), high transition / obs entropy"
# )
# plt.savefig("img/smc_high_entropy_hmm_quad_plot.png")
#
# config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.1, 0.9)
# key, fig = run_hmm_quad_plot(key, gold_standard, config)
# fig.suptitle(
#    "Discrete HMM ($\#$ latent states = 50, $\#$ obs. states = 50, T = 50), low transition / high obs entropy"
# )
# plt.savefig("img/smc_mixed_entropy_hmm_quad_plot.png")
#
# config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.1, 0.1)
# key, fig = run_hmm_quad_plot(key, gold_standard, config)
# fig.suptitle(
#    "Discrete HMM ($\#$ latent states = 50, $\#$ obs. states = 50, T = 50), low transition / obs entropy"
# )
# plt.savefig("img/smc_low_entropy_hmm_quad_plot.png")
