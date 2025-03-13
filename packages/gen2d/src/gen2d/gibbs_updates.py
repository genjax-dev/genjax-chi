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

from typing import Any

import gen2d.conjugacy as conjugacy
import jax
import jax.numpy as jnp
import gen2d.model_simple_continuous as model_simple_continuous
import tensorflow_probability.substrates.jax as tfp
import gen2d.utils as utils

import genjax
from genjax import ChoiceMapBuilder as C


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
    new_means = tfp.distributions.Normal(
        loc=posterior_means, scale=posterior_variances
    ).sample(seed=key)
    chosen_means = utils.mywhere(category_counts == 0, current_means, new_means)
    return chosen_means


def update_xy_mean(key, tracediff):
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
    ) = utils.markov_for_xy_mean_from_trace(tracediff)

    category_counts = utils.category_count(datapoint_indexes, n_clusters)
    cluster_means = utils.compute_means(
        datapoints, datapoint_indexes, n_clusters, category_counts
    )

    posterior_means, posterior_variances = conjugacy.update_normal_normal_conjugacy(
        prior_mean, current_variance, cluster_means, obs_variance, category_counts
    )

    new_means = mean_resampling(
        key, posterior_means, posterior_variances, current_means, category_counts
    )

    # Update tracediff
    new_tracediff = utils.concat(tracediff, C["blob_model", "xy_mean"].set(new_means))
    return new_tracediff


def update_rgb_mean(key, tracediff):
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
    ) = utils.markov_for_rgb_mean_from_trace(tracediff)

    category_counts = utils.category_count(datapoint_indexes, n_clusters)
    cluster_means = utils.compute_means(
        datapoints, datapoint_indexes, n_clusters, category_counts
    )

    posterior_means, posterior_variances = conjugacy.update_normal_normal_conjugacy(
        prior_mean, current_variance, cluster_means, obs_variance, category_counts
    )

    new_means = mean_resampling(
        key, posterior_means, posterior_variances, current_means, category_counts
    )

    # Update tracediff
    new_tracediff = utils.concat(tracediff, C["blob_model", "rgb_mean"].set(new_means))
    return new_tracediff


def update_cluster_assignment(key, tracediff):
    """Perform Gibbs update for cluster assignments of each datapoint.

    Vectorized implementation that computes all local densities in parallel.
    """
    # Extract all needed parameters once
    n_clusters = tracediff.args[0].n_blobs
    n_datapoints = tracediff.args[0].H * tracediff.args[0].W

    # Get all datapoints at once
    datapoints_xy = tracediff.chm["likelihood_model", "xy"]
    datapoints_rgb = tracediff.chm["likelihood_model", "rgb"]

    # Get cluster parameters
    cluster_xy_means = tracediff.chm["blob_model", "xy_mean"]
    cluster_xy_spread = tracediff.chm["blob_model", "sigma_xy"]
    cluster_rgb_means = tracediff.chm["blob_model", "rgb_mean"]
    cluster_rgb_spread = tracediff.chm["blob_model", "sigma_rgb"]
    mixture_weights = tracediff.chm["blob_model", "mixture_weight"]
    mixture_probs = mixture_weights / jnp.sum(mixture_weights)

    likelihood_params = model_simple_continuous.LikelihoodParams(
        cluster_xy_means,
        cluster_xy_spread,
        cluster_rgb_means,
        cluster_rgb_spread,
        mixture_probs,
    )

    # Vectorized computation across all points and clusters
    def compute_density_for_point(x_idx):
        chm = C["xy"].set(datapoints_xy[x_idx]).at["rgb"].set(datapoints_rgb[x_idx])
        return jax.vmap(
            lambda i: model_simple_continuous.likelihood_model.assess(
                chm.at["blob_idx"].set(i), (i, likelihood_params)
            )[0]
        )(jnp.arange(n_clusters))

    local_densities = jax.vmap(compute_density_for_point)(jnp.arange(n_datapoints))

    # Sample new assignments
    new_datapoint_indexes = tfp.distributions.Categorical(
        logits=local_densities
    ).sample(seed=key)

    # Update tracediff
    new_tracediff = utils.concat(
        tracediff, C["likelihood_model", "blob_idx"].set(new_datapoint_indexes)
    )
    return new_tracediff


def update_mixture_weight(key, tracediff):
    """Perform Gibbs update for the mixture weights of the Gaussian components.

    This function uses Dirichlet-categorical conjugacy to update the mixture weights
    by:
    1. Computing counts of datapoints per cluster
    2. Adding prior alpha to get posterior Dirichlet parameters
    3. Sampling new weights from posterior Dirichlet

    Args:
        key: JAX random key
        trace: Current execution trace containing model state

    Returns:
        Updated trace with new mixture weights
    """
    n_clusters = tracediff.args[0].n_blobs
    prior_alpha = tracediff.args[0].alpha
    datapoint_indexes = tracediff.chm["likelihood_model", "blob_idx"]
    category_counts = utils.category_count(datapoint_indexes, n_clusters)

    # TODO: check math here. might be alpha/n or something.
    # check the way George did it.
    # this seems to currently update the mixture weight correctly though.
    new_alphas = prior_alpha + category_counts

    new_weights = genjax.dirichlet.sample(key, new_alphas)

    # Update tracediff
    new_tracediff = utils.concat(
        tracediff, C["blob_model", "mixture_weight"].set(new_weights)
    )
    return new_tracediff


def update_xy_sigma(key, tracediff):
    """Perform Gibbs update for the spatial variance parameters of each cluster.

    Uses inverse-gamma conjugate prior to update sigma_xy based on:
    1. Prior parameters a_xy, b_xy from hyperparameters
    2. Empirical means and counts of points in each cluster
    3. Posterior parameters derived from conjugate update equations

    Args:
        key: JAX random key
        trace: Current execution trace containing model state

    Returns:
        Updated trace with new sigma_xy values
    """
    # Get data and parameters from trace
    datapoint_indexes = tracediff.chm["likelihood_model", "blob_idx"]
    datapoints: Any = tracediff.chm["likelihood_model", "xy"]
    cluster_means = tracediff.chm["blob_model", "xy_mean"]
    n_clusters = tracediff.args[0].n_blobs
    prior_alphas = tracediff.args[0].a_xy
    prior_betas = tracediff.args[0].b_xy

    # Calculate posterior parameters using conjugate update function
    category_counts = utils.category_count(datapoint_indexes, n_clusters)
    squared_deviations = utils.compute_squared_deviations(
        datapoints, datapoint_indexes, cluster_means, n_clusters
    )
    posterior_alphas, posterior_betas = conjugacy.update_inverse_gamma_normal_conjugacy(
        prior_alphas, prior_betas, squared_deviations, category_counts
    )

    # Sample new sigma values from inverse gamma posterior
    key, subkey = jax.random.split(key)
    new_sigma_xy = tfp.distributions.InverseGamma(
        concentration=posterior_alphas, scale=posterior_betas
    ).sample(seed=subkey)

    # Rescaling sigma^2 -> sigma
    new_sigma_xy = jnp.sqrt(new_sigma_xy)

    # Update tracediff
    new_tracediff = utils.concat(
        tracediff, C["blob_model", "sigma_xy"].set(new_sigma_xy)
    )

    return new_tracediff


# TODO: currently absolutely busted. And very slow.
def update_rgb_sigma(key, tracediff):
    """Perform Gibbs update for the RGB variance parameters of each cluster.

    Uses inverse-gamma conjugate prior to update sigma_rgb based on:
    1. Prior parameters a_rgb, b_rgb from hyperparameters
    2. Empirical means and counts of points in each cluster
    3. Posterior parameters derived from conjugate update equations

    Args:
        key: JAX random key
        trace: Current execution trace containing model state

    Returns:
        Updated trace with new sigma_rgb values
    """
    # Get data and parameters from trace
    datapoint_indexes = tracediff.chm["likelihood_model", "blob_idx"]
    datapoints = tracediff.chm["likelihood_model", "rgb"]
    cluster_means = tracediff.chm["blob_model", "rgb_mean"]
    n_clusters = tracediff.args[0].n_blobs
    prior_alphas = tracediff.args[0].a_rgb
    prior_betas = tracediff.args[0].b_rgb

    # Calculate posterior parameters using conjugate update function
    category_counts = utils.category_count(datapoint_indexes, n_clusters)
    squared_deviations = utils.compute_squared_deviations(
        datapoints, datapoint_indexes, cluster_means, n_clusters
    )
    posterior_alphas, posterior_betas = conjugacy.update_inverse_gamma_normal_conjugacy(
        prior_alphas, prior_betas, squared_deviations, category_counts
    )

    # Sample new sigma values from inverse gamma posterior
    key, subkey = jax.random.split(key)
    new_sigma_rgb = tfp.distributions.InverseGamma(
        concentration=posterior_alphas, scale=posterior_betas
    ).sample(seed=subkey)

    # Rescaling sigma^2 -> sigma
    new_sigma_rgb = jnp.sqrt(new_sigma_rgb)

    # jax.debug.print("datapoint_indexes sample: {x}", x=datapoint_indexes[:5])
    # jax.debug.print("datapoints sample: {x}", x=datapoints[:5])
    # jax.debug.print("cluster_means sample: {x}", x=cluster_means[:5])
    # jax.debug.print("prior_alphas: {x}", x=prior_alphas)
    # jax.debug.print("prior_betas: {x}", x=prior_betas)
    # jax.debug.print("category_counts: {x}", x=category_counts)
    # jax.debug.print("squared_deviations shape: {x}", x=squared_deviations.shape)
    # jax.debug.print("squared_deviations sample: {x}", x=squared_deviations[:5])
    # jax.debug.print("posterior_alphas sample: {x}", x=posterior_alphas[:5])
    # jax.debug.print("posterior_betas sample: {x}", x=posterior_betas[:5])
    # jax.debug.print("new_sigma_rgb sample: {x}", x=new_sigma_rgb[:5])

    # Update tracediff
    new_tracediff = utils.concat(
        tracediff, C["blob_model", "sigma_rgb"].set(new_sigma_rgb)
    )

    return new_tracediff
