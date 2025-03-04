### This file contains the different Gibbs updates for the Gen2D model.
#
# The inference logic is simply block-Gibbs:
# - sample an initial trace:
# - perform N Gibbs sweeps on that trace
# - return the posterior sample
#
# Initialization:
# - preprocess a H by W image into a (H*W, 5) array of (x, y, r, g, b) values
# - generate an initial trace with constraints so that every Gaussian ends up associated with at least one datapoint.

# Gibbs sweep:
# - update xy_mean, xy_sigma, rgb_mean, rgb_sigma in parallel using Normal-Inverse-Gamma conjugacy, given the datapoints currently associated with it
# - update cluster assignment for each datapoint in parallel via enumerative Gibbs
# - update mixture weights using Dirichlet categorical conjugacy, using the fact that normalizing many gamma-distributed mixture weights is the same as sampling from a Dirichlet distribution

import conjugacy
import jax.numpy as jnp
import model_simple_continuous
import utils

import genjax
from genjax import ChoiceMapBuilder as C


# Compute the means of the datapoints per cluster
# Will contain some NaN due to clusters having no datapoint
def compute_means(datapoints, datapoint_indexes, n_clusters, category_counts):
    # TODO: busted
    # return jax.vmap(
    #     lambda i: jnp.sum(
    #         jnp.where(
    #             jnp.expand_dims(datapoint_indexes == i, -1),
    #             datapoints,
    #             jnp.zeros_like(datapoints),
    #         ),
    #         axis=0,
    #     ),
    #     in_axes=(0),
    #     out_axes=(0),
    # )(jnp.arange(n_clusters)) / jnp.expand_dims(category_counts, -1)
    return None


# Gibbs resampling of cluster means
def xy_mean_resampling(key, hypers, current_means, prior_variance, category_counts):
    args = (jnp.arange(hypers.n_blobs), hypers)
    obs = C["sigma_xy"].set(prior_variance)
    new_means = (
        model_simple_continuous.xy_model.vmap(in_axes=(0, None))
        .importance(key, obs, args)[0]
        .get_choices()["xy_mean"]
    )

    # Remove the sampled Nan due to clusters having no datapoint and pick previous mean in that case, i.e. no Gibbs update for them
    chosen_means = jnp.where(category_counts == 0, current_means, new_means)

    return chosen_means


def update_xy_mean(key, trace):
    (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        current_means,
        prior_variance,
        obs_variance,
    ) = utils.markov_for_xy_mean_from_trace(trace)

    category_counts = utils.category_count(datapoint_indexes, n_clusters)

    cluster_means = compute_means(
        datapoints, datapoint_indexes, n_clusters, category_counts
    )

    posterior_means, posterior_variances = conjugacy.update_normal_normal_conjugacy(
        prior_mean, prior_variance, cluster_means, obs_variance, category_counts
    )

    key, subkey = jax.random.split(key)
    new_means = xy_mean_resampling(
        key, hypers, current_means, category_counts, prior_variance, category_counts
    )

    new_trace = utils.update_trace_with_xy_mean(subkey, trace, new_means)

    return new_trace


def update_xy_sigma(key, trace):
    (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        current_means,
        prior_variance,
        obs_variance,
    ) = utils.markov_for_xy_mean_from_trace(trace)

    prior_alphas = jnp.tile(trace.get_args()[0].a_xy, (n_clusters, 1))
    posterior_alphas = prior_alphas + jnp.expand_dims(n_clusters, -1) / 2

    prior_betas = jnp.tile(trace.get_args()[0].b_xy, (n_clusters, 1))
    empirical_cluster_means = compute_means(
        datapoints, datapoint_indexes, n_clusters, category_counts
    )
    mean_diff = 1 / 2 * (empirical_cluster_means - prior_mean)
    posterior_betas = prior_betas + mean_diff
    new_sigma_xy = genjax.inverse_gamma.vmap().vmap()(posterior_alphas, posterior_betas)

    sigma_xy = trace.get_choices()["blob_model", "sigma_xy"]
    new_sigma_xy = jnp.where(category_counts == 0, sigma_xy, new_sigma_xy)

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        key, C["blob_model", "sigma_xy"].set(new_sigma_xy), argdiffs
    )
    return new_trace


def update_rgb_mean(key, trace):
    return trace


def update_rgb_sigma(key, trace):
    return trace


def update_cluster_assignment(key, trace):
    def compute_local_density(x, i):
        datapoint_xy_mean = trace.get_choices()["likelihood_model", "xy", x]
        datapoint_rgb_mean = trace.get_choices()["likelihood_model", "rgb", x]

        chm = (
            C["xy"]
            .set(datapoint_mean)
            .at["blob_idx"]
            .set(i)
            .at["rgb"]
            .set(datapoint_rgb_mean)
        )

        # TODO: from here
        clusters = Cluster(trace.get_choices()["clusters", "mean"])
        probs = trace.get_choices()["probs"]
        args = (i, probs, clusters)
        model_logpdf, _ = likelihood_model.assess(chm, args)
        return model_logpdf

    local_densities = jax.vmap(
        lambda x: jax.vmap(lambda i: compute_local_density(x, i))(
            jnp.arange(n_clusters)
        )
    )(jnp.arange(n_datapoints))

    key, subkey = jax.random.split(key)
    new_datapoint_indexes = (
        genjax.categorical.vmap().simulate(key, (local_densities,)).get_choices()
    )

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["likelihood_model", "idx"].set(new_datapoint_indexes), argdiffs
    )
    return new_trace


def update_mixture_weight(key, trace):
    n_clusters = trace.get_args()[0].n_blobs
    prior_alpha = trace.get_args()[0].alpha
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    category_counts = category_count(datapoint_indexes, n_clusters)

    new_alphas = prior_alpha + category_counts
    key, subkey = jax.random.split(key)
    new_weights = genjax.dirichlet.sample(key, new_alphas)

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["blob_model", "mixture_weight"].set(new_weights), argdiffs
    )

    return new_trace
