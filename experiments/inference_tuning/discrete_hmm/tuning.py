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
config = genjax.DiscreteHMMConfiguration.new(50, 1, 1, 0.3, 0.1)

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
# SDOS
#####


def run_sdos_experiment(key):
    def sdos_visualizer(fig, axes, df):
        pal = "Set2"
        dx, dy = "particles", "Model average symmetric KL estimate"
        axes = pt.half_violinplot(
            x=dx,
            y=dy,
            data=df,
            palette=pal,
            bw=0.2,
            cut=0.0,
            scale="area",
            width=2.0,
            inner=None,
        )
        axes = sns.stripplot(
            x=dx,
            y=dy,
            data=df,
            palette=pal,
            edgecolor="white",
            size=3,
            jitter=1,
            zorder=0,
        )
        axes = sns.boxplot(
            x=dx,
            y=dy,
            data=df,
            color="black",
            width=0.15,
            zorder=10,
            showcaps=True,
            boxprops={"facecolor": "none", "zorder": 10},
            showfliers=False,
            whiskerprops={"linewidth": 2, "zorder": 10},
            saturation=1,
        )

    def sdos_for_nparticles(key, custom_smc):
        inf_selection = genjax.Selection([("z", "latent")])
        key, ratio, (fwd, bwd) = genjax.sdos(
            hidden_markov_model,
            custom_smc,
            inf_selection,
            1,
            200,
        )(key, (num_steps, config))
        return key, (jnp.mean(ratio), jnp.sqrt(jnp.var(ratio))), ratio

    def sdos_plot(key, make_custom_smc):
        fig, axes = plt.subplots(dpi=400)
        d = {"particles": [], "Model average symmetric KL estimate": []}
        df = pd.DataFrame(d)
        custom_smcs = list(
            map(make_custom_smc, [1, 2, 5, 10, 20, 50, 100, 200, 500])
        )

        for custom_smc in track(custom_smcs, description="Real runs"):
            ratio = np.array([], dtype=np.float32)

            def _lambda(key, custom_smc):
                key, *sub_keys = jax.random.split(key, 10 + 1)
                sub_keys = jnp.array(sub_keys)
                _, (_, _), r = jax.vmap(sdos_for_nparticles, in_axes=(0, None))(
                    sub_keys, custom_smc
                )
                return key, r

            jitted = jax.jit(_lambda)
            for _ in range(0, 50):
                key, r = jitted(key, custom_smc)
                ratio = np.append(ratio, np.array(r))
            dx = np.repeat(custom_smc.num_particles, len(ratio))
            d = {"particles": dx, "Model average symmetric KL estimate": ratio}
            new = pd.DataFrame(data=d)
            df = pd.concat([df, new], ignore_index=True)

        sdos_visualizer(fig, axes, df)

        labels_handles = {
            label: handle
            for handle, label in zip(*axes.get_legend_handles_labels())
        }

        fig.legend(
            labels_handles.values(),
            labels_handles.keys(),
            loc="upper right",
        )
        return key, fig

    # Run SMC with prior as proposal.
    key, fig1 = sdos_plot(key, custom_smc_with_prior)
    fig1.suptitle("SDOS (Proposal is prior)")
    plt.xlabel("# of particles")
    plt.ylabel("Estimator")
    plt.savefig("img/sdos_prior_proposal.png")

    # Run SMC with transition as proposal.
    key, fig2 = sdos_plot(key, custom_smc_with_transition)
    fig2.suptitle("SDOS (Data-driven proposal)")
    plt.xlabel("# of particles")
    plt.ylabel("Estimator")
    plt.savefig("img/sdos_transition_proposal.png")
    return key


#####
# AIDE
#####


def run_aide_experiment(key):
    def aide_visualizer(fig, axes, df):
        pal = "Set2"
        dx, dy = "logpdf", "est"
        axes = pt.RainCloud(
            x=dx,
            y=dy,
            data=df,
            bw=0.2,
            scale="area",
            width_viol=1.0,
            box_showfliers=False,
        )
        axes.set_xticks([])

    @genjax.gen(
        prox.ChoiceMapDistribution,
        selection=genjax.AllSelection(),
    )
    def exact_hmm_posterior(key, config, observations):
        key, v = genjax.trace(("z", "latent"), genjax.DiscreteHMM)(
            key, (config, observations)
        )
        return (key,)

    def sample_and_score(key, config):
        selection = genjax.Selection([("z", "observation")])
        key, tr = genjax.simulate(hidden_markov_model)(key, (num_steps, config))
        chm = tr.get_choices().strip_metadata()
        observations = chm["z", "observation"]
        logprob = genjax.DiscreteHMM.data_logpdf(config, observations)
        chm, _ = selection.filter(chm)
        return key, (logprob, chm)

    def aide_plot(key, custom_smc):
        fig, axes = plt.subplots(figsize=(14, 14), dpi=400)
        axes.set_xticks([])
        d = {"logpdf": [], "est": []}
        df = pd.DataFrame(d)
        ratio = np.array([], dtype=np.float32)
        exact_score = np.array([], dtype=np.float32)

        def _lambda(key, custom_smc):
            key, (exact_score, chm) = sample_and_score(key, config)
            final_target = prox.Target(
                hidden_markov_model,
                None,
                (num_steps, config),
                chm,
            )
            observations = chm["z", "observation"]
            key, *sub_keys = jax.random.split(key, 250 + 1)
            sub_keys = jnp.array(sub_keys)
            _, est, (_, _) = jax.vmap(
                genjax.aide(exact_hmm_posterior, custom_smc, 1, 250),
                in_axes=(0, None, None),
            )(sub_keys, (config, observations), (final_target,))
            exact_score = jnp.repeat(exact_score, len(est))
            return key, (exact_score, est)

        jitted = jax.jit(_lambda)
        for _ in track(range(0, 10), description="Sample + AIDE"):
            key, (es, r) = jitted(key, custom_smc)
            ratio = np.append(ratio, np.array(r))
            exact_score = np.append(exact_score, np.array(es))
        d = {
            "logpdf": exact_score,
            "est": ratio,
        }
        new = pd.DataFrame(data=d)
        df = pd.concat([df, new], ignore_index=True)

        aide_visualizer(fig, axes, df)

        labels_handles = {
            label: handle
            for handle, label in zip(*axes.get_legend_handles_labels())
        }

        fig.legend(
            labels_handles.values(),
            labels_handles.keys(),
            loc="upper right",
        )
        return key, fig

    # Run SMC with prior as proposal.
    key, fig = aide_plot(key, custom_smc_with_prior(10))
    fig.suptitle("AIDE (10 particle, proposal is prior) vs. exact")
    plt.xlabel("Log data marginal")
    plt.ylabel("Estimator")
    plt.savefig("img/10_aide_prior_proposal.png")

    key, fig = aide_plot(key, custom_smc_with_prior(100))
    fig.suptitle("AIDE (100 particle, proposal is prior) vs. exact")
    plt.xlabel("Log data marginal")
    plt.ylabel("Estimator")
    plt.savefig("img/100_aide_prior_proposal.png")

    # Run SMC with transition as proposal.
    key, fig = aide_plot(key, custom_smc_with_transition(10))
    fig.suptitle("AIDE (10 particle, data-driven proposal) vs. exact")
    plt.xlabel("Log data marginal")
    plt.ylabel("Estimator")
    plt.savefig("img/10_aide_transition_proposal.png")

    key, fig = aide_plot(key, custom_smc_with_transition(100))
    fig.suptitle("AIDE (100 particle, data-driven proposal) vs. exact")
    plt.xlabel("Log data marginal")
    plt.ylabel("Estimator")
    plt.savefig("img/100_aide_transition_proposal.png")

    return key


key = run_aide_experiment(key)
