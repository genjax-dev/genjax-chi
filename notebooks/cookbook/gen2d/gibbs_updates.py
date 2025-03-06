"""
Gibbs sampling updates for the Gen2D model.

This module implements block Gibbs sampling for a 2D Gaussian mixture model with RGB values.
The inference procedure alternates between:

1. Updating cluster parameters:
   - xy_mean, xy_sigma: Location and scale of 2D Gaussians using Normal-Inverse-Gamma conjugacy
   - rgb_mean, rgb_sigma: Color parameters using Normal-Inverse-Gamma conjugacy

2. Updating cluster assignments:
   - For each datapoint, sample new cluster via enumerative Gibbs
   - Uses parallel updates across all points

3. Updating mixture weights:
   - Uses Dirichlet-categorical conjugacy
   - Leverages fact that normalized Gamma RVs follow Dirichlet distribution

The model is initialized by preprocessing an HxW image into (x,y,r,g,b) points and
generating an initial trace where each Gaussian is associated with at least one point.
"""

import conjugacy
import jax
import jax.numpy as jnp
import utils

import genjax
from genjax import ChoiceMapBuilder as C


# Compute the means of the datapoints per cluster
# Will contain some NaN due to clusters having no datapoint
def compute_means(datapoints, datapoint_indexes, n_clusters, category_counts):
    """Compute the mean of datapoints for each cluster.

    Args:
        datapoints: Array of shape (N, 2) containing 2D coordinates
        datapoint_indexes: Array of shape (N,) containing cluster assignments
        n_clusters: Integer number of clusters
        category_counts: Array of shape (n_clusters,) containing counts per cluster

    Returns:
        Array of shape (n_clusters, 2) containing mean coordinates per cluster
    """
    # First sum up all points belonging to each cluster
    sums = jax.vmap(
        lambda i: jnp.sum(
            jnp.where(
                datapoint_indexes[:, None]
                == i,  # Broadcasting to match datapoints shape
                datapoints,
                0.0,
            ),
            axis=0,  # Sum over datapoints
        )
    )(jnp.arange(n_clusters))

    # Divide by counts to get means, handling division by zero
    means = sums / category_counts[:, None]  # Broadcasting counts for division
    return means


def mean_resampling(
    key, posterior_means, posterior_variances, current_means, category_counts
):
    """Perform Gibbs resampling of cluster means.

    Args:
        key: JAX random key
        posterior_means: Array of shape (n_clusters, 2) containing posterior means
        posterior_variances: Array of shape (n_clusters, 2) containing posterior variances
        current_means: Array of shape (n_clusters, 2) containing current cluster means
        category_counts: Array of shape (n_clusters,) containing counts per cluster

    Returns:
        Array of shape (n_clusters, 2) containing updated cluster means, where clusters
        with no datapoints retain their previous means
    """
    new_means = (
        genjax.normal.vmap(in_axes=(0, 0))
        .simulate(key, (posterior_means, posterior_variances))
        .get_retval()
    )
    chosen_means = utils.mywhere(category_counts == 0, current_means, new_means)
    return chosen_means


def update_xy_mean(key, trace):
    """Perform Gibbs update for the spatial (xy) means of each Gaussian component.

    This function:
    1. Extracts relevant data from the trace
    2. Computes cluster assignments and means
    3. Updates the means using normal-normal conjugacy
    4. Resamples new means from the posterior
    5. Updates the trace with new means

    Args:
        key: JAX random key for sampling
        trace: GenJAX trace containing current model state

    Returns:
        Updated trace with new xy_mean values
    """
    (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        current_means,
        current_variance,
        obs_variance,
    ) = utils.markov_for_xy_mean_from_trace(trace)

    category_counts = utils.category_count(datapoint_indexes, n_clusters)
    cluster_means = compute_means(
        datapoints, datapoint_indexes, n_clusters, category_counts
    )

    posterior_means, posterior_variances = conjugacy.update_normal_normal_conjugacy(
        prior_mean, current_variance, cluster_means, obs_variance, category_counts
    )

    new_means = mean_resampling(
        key, posterior_means, posterior_variances, current_means, category_counts
    )

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        key, C["blob_model", "xy_mean"].set(new_means), argdiffs
    )

    return new_trace


def update_rgb_mean(key, trace):
    """Perform Gibbs update for the RGB means of each Gaussian component.

    This function:
    1. Extracts relevant data from the trace
    2. Computes cluster assignments and means
    3. Updates the means using normal-normal conjugacy
    4. Resamples new means from the posterior
    5. Updates the trace with new means

    Args:
        key: JAX random key for sampling
        trace: GenJAX trace containing current model state

    Returns:
        Updated trace with new rgb_mean values
    """
    (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        current_means,
        current_variance,
        obs_variance,
    ) = utils.markov_for_rgb_mean_from_trace(trace)

    category_counts = utils.category_count(datapoint_indexes, n_clusters)
    cluster_means = compute_means(
        datapoints, datapoint_indexes, n_clusters, category_counts
    )

    posterior_means, posterior_variances = conjugacy.update_normal_normal_conjugacy(
        prior_mean, current_variance, cluster_means, obs_variance, category_counts
    )

    new_means = mean_resampling(
        key, posterior_means, posterior_variances, current_means, category_counts
    )

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        key, C["blob_model", "rgb_mean"].set(new_means), argdiffs
    )

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
