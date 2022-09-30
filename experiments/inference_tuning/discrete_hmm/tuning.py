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
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
from rich.progress import track


# Global setup.
sns.set()
key = jax.random.PRNGKey(314159)
num_steps = 50
config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.8, 0.5)

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
        width=0.6,
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
        showfliers=True,
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
        100,
    )(key, (num_steps, config))
    return key, (jnp.mean(ratio), jnp.sqrt(jnp.var(ratio))), ratio


def sdos_plot(key, make_custom_smc):
    fig, axes = plt.subplots(figsize=(14, 14), dpi=200)
    d = {"particles": [], "Model average symmetric KL estimate": []}
    df = pd.DataFrame(d)
    custom_smcs = list(
        map(make_custom_smc, [1, 2, 5, 10, 20, 50, 100, 200, 500])
    )
    jitted = jax.jit(sdos_for_nparticles)

    # Warmup.
    for custom_smc in track(custom_smcs, description="Warm up"):
        key, (mean, var), ratio = jitted(key, custom_smc)

    # Real runs.
    for custom_smc in track(custom_smcs, description="Real runs"):
        key, *sub_keys = jax.random.split(key, 50 + 1)
        sub_keys = jnp.array(sub_keys)
        _, (mean, var), ratio = jax.jit(jax.vmap(jitted, in_axes=(0, None)))(
            sub_keys, custom_smc
        )
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
        fontsize=20,
    )
    return key, fig


# Run SMC with prior as proposal.
key, fig1 = sdos_plot(key, custom_smc_with_prior)
fig1.suptitle("SDOS (Proposal is prior)", fontsize=24)
fig1.savefig("sdos_prior_proposal.png")

# Run SMC with transition as proposal.
key, fig2 = sdos_plot(key, custom_smc_with_transition)
fig2.suptitle("SDOS (Data-driven proposal)", fontsize=24)
plt.savefig("sdos_transition_proposal.png")


#####
# AIDE
#####


# @genjax.gen(
#    prox.ChoiceMapDistribution,
#    selection=genjax.AllSelection(),
# )
# def exact_hmm_posterior(key, config, observations):
#    key, v = genjax.trace(("z", "latent"), ExactHMMPosterior)(
#        key, (config, observations)
#    )
#    return (key,)
#
#
# def sample_and_score(key, config):
#    selection = genjax.Selection([("z", "observation")])
#    key, tr = genjax.simulate(hidden_markov_model)(key, (num_steps, config))
#    chm = tr.get_choices().strip_metadata()
#    observations = chm["z", "observation"]
#    logprob = ExactHMMPosterior.data_logpdf(config, observations)
#    chm, _ = selection.filter(chm)
#    return key, (logprob, chm)
#
#
# def random_aide(key, config, custom_smc):
#    key, (_, chm) = sample_and_score(key, config)
#    final_target = prox.Target(
#        hidden_markov_model,
#        None,
#        (num_steps, config),
#        chm,
#    )
#    observations = chm["z", "observation"]
#    key, *sub_keys = jax.random.split(key, 500 + 1)
#    sub_keys = jnp.array(sub_keys)
#    _, est, (logpq, logqp) = jax.vmap(
#        genjax.aide(exact_hmm_posterior, custom_smc, 1, 100),
#        in_axes=(0, None, None),
#    )(sub_keys, (config, observations), (final_target,))
#    return key, (jnp.mean(est), jnp.sqrt(jnp.var(est))), (logpq, logqp)
#
#
# key = jax.random.PRNGKey(314159)
# custom_smc = genjax.CustomSMC(
#    meta_initial_position,
#    hmm_meta_next_target,
#    transition_proposal,
#    lambda _: num_steps,
#    5,
# )
# key, est, (logpq, logqp) = jax.jit(random_aide)(key, config, custom_smc)
# print(est)
