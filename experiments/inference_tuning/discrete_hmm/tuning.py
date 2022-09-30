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
from model_config import DiscreteHMMConfiguration, hidden_markov_model
from inference_config import (
    meta_initial_position,
    hmm_meta_next_target,
    transition_proposal,
)
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Global setup.
key = jax.random.PRNGKey(314159)
num_steps = 50
config = DiscreteHMMConfiguration.new(50, 2, 1, 0.8, 0.5)

#####
# SDOS
#####


def sdos_for_nparticles(key, n):
    inf_selection = genjax.Selection([("z", "latent")])
    custom_smc = genjax.CustomSMC(
        meta_initial_position,
        hmm_meta_next_target,
        transition_proposal,
        lambda _: num_steps,
        n,
    )
    key, *sub_keys = jax.random.split(key, 500 + 1)
    sub_keys = jnp.array(sub_keys)
    _, ratio, (fwd, bwd) = jax.vmap(
        genjax.sdos(
            hidden_markov_model,
            custom_smc,
            inf_selection,
            1,
            200,
        ),
        in_axes=(0, None),
    )(sub_keys, (num_steps, config))
    return key, (jnp.mean(ratio), jnp.sqrt(jnp.var(ratio))), ratio


def sdos_plot(key, make_custom_smc):
    fig, axes = plt.subplots(figsize=(14, 14), dpi=200)

    for n_particles in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        print(n_particles)
        custom_smc = make_custom_smc(n_particles)
        key, (mean, var), ratio = jax.jit(
            sdos_for_nparticles, static_argnums=1
        )(key, n_particles)
        sdos_visualizer(
            fig,
            axes,
            n_particles,
            ratio,
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
        fontsize=20,
    )
    return key, fig


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
