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
import model_simple_continuous
import utils

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
    cluster_means = utils.compute_means(
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
    cluster_means = utils.compute_means(
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


def update_cluster_assignment(key, trace):
    """Perform Gibbs update for cluster assignments of each datapoint.

    Vectorized implementation that computes all local densities in parallel.
    """
    # Extract all needed parameters once
    n_clusters = trace.args[0].n_blobs
    n_datapoints = trace.args[0].H * trace.args[0].W

    # Get all datapoints at once
    datapoints_xy = trace.get_choices()["likelihood_model", "xy"]
    datapoints_rgb = trace.get_choices()["likelihood_model", "rgb"]

    # Get cluster parameters
    cluster_xy_means = trace.get_choices()["blob_model", "xy_mean"]
    cluster_xy_spread = trace.get_choices()["blob_model", "sigma_xy"]
    cluster_rgb_means = trace.get_choices()["blob_model", "rgb_mean"]
    cluster_rgb_spread = trace.get_choices()["blob_model", "sigma_rgb"]
    mixture_weights = trace.get_choices()["blob_model", "mixture_weight"]
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
    key, subkey = jax.random.split(key)
    new_datapoint_indexes = (
        genjax.categorical.vmap().simulate(key, (local_densities,)).get_choices()
    )

    # Update trace
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["likelihood_model", "blob_idx"].set(new_datapoint_indexes), argdiffs
    )
    return new_trace


def update_mixture_weight(key, trace):
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
    n_clusters = trace.args[0].n_blobs
    prior_alpha = trace.args[0].alpha
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    category_counts = utils.category_count(datapoint_indexes, n_clusters)

    # TODO: check math here. might be alpha/n or something.
    # check the way George did it.
    # this seems to currently update the mixture weight correctly though.
    new_alphas = prior_alpha + category_counts
    key, subkey = jax.random.split(key)
    new_weights = genjax.dirichlet.sample(key, new_alphas)

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["blob_model", "mixture_weight"].set(new_weights), argdiffs
    )

    return new_trace


def update_xy_sigma(key, trace):
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
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    datapoints = trace.get_choices()["likelihood_model", "xy"]
    cluster_means = trace.get_choices()["blob_model", "xy_mean"]
    n_clusters = trace.args[0].n_blobs
    prior_alphas = trace.args[0].a_xy
    prior_betas = trace.args[0].b_xy

    # Get counts per cluster
    category_counts = utils.category_count(datapoint_indexes, n_clusters)

    # Compute sum of squared deviations using cluster means (μ), not empirical means
    squared_deviations = utils.compute_squared_deviations(
        datapoints, datapoint_indexes, cluster_means, n_clusters
    )

    # TODO: this is a hack breaking proper Bayesian update. I don't understand why I need this yet. There's probably a proper bug somewhere.
    scale = jnp.array([1000.0 * 700.0]) / 100.0  # Total area / target variance
    squared_deviations = squared_deviations / scale

    # Calculate posterior parameters using conjugate update function
    posterior_alphas, posterior_betas = conjugacy.update_inverse_gamma_normal_conjugacy(
        prior_alphas, prior_betas, squared_deviations, category_counts
    )

    # jax.debug.print("Cluster means: {m}", m=cluster_means)

    # # Print stats for first cluster using where
    # mask = (datapoint_indexes == 0)[:, None]
    # diffs_0 = jnp.where(
    #     mask,
    #     datapoints - cluster_means[0],
    #     0.0
    # )
    # sq_diffs_0 = jnp.sum(diffs_0**2, axis=0)

    # jax.debug.print("Cluster 0 count: {n}", n=jnp.sum(datapoint_indexes == 0))
    # jax.debug.print("Cluster 0 mean: {m}", m=cluster_means[0])
    # jax.debug.print("Cluster 0 sum sq diffs: {s}", s=sq_diffs_0)
    # jax.debug.print("All squared deviations: {s}", s=squared_deviations[:10])
    # jax.debug.print("Category counts: {c}", c=category_counts[:10])
    # jax.debug.print("Prior alpha: {a}", a=prior_alphas[:10])
    # jax.debug.print("Prior beta: {b}", b=prior_betas[:10])

    # Sample new sigma values from inverse gamma posterior
    key, subkey = jax.random.split(key)
    new_sigma_xy = (
        genjax.inverse_gamma.vmap(in_axes=(0, 0))
        .simulate(key, (posterior_alphas, posterior_betas))
        .get_retval()
    )
    # Keep old sigma values for empty clusters
    old_sigma_xy = trace.get_choices()["blob_model", "sigma_xy"]
    # jax.debug.print("Old sigma_xy: {s}", s=old_sigma_xy[:10])
    # jax.debug.print("New sigma_xy (before empty cluster fix): {s}", s=new_sigma_xy[:10])
    new_sigma_xy = jnp.where(category_counts[:, None] == 0, old_sigma_xy, new_sigma_xy)
    # jax.debug.print("Final sigma_xy: {s}", s=new_sigma_xy[:10])

    # Update trace with new sigma values
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["blob_model", "sigma_xy"].set(new_sigma_xy), argdiffs
    )
    return new_trace


def update_rgb_sigma(key, trace):
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
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    datapoints = trace.get_choices()["likelihood_model", "rgb"]
    cluster_means = trace.get_choices()["blob_model", "rgb_mean"]
    n_clusters = trace.args[0].n_blobs
    prior_alphas = trace.args[0].a_rgb
    prior_betas = trace.args[0].b_rgb

    # Get counts per cluster
    category_counts = utils.category_count(datapoint_indexes, n_clusters)

    # Compute sum of squared deviations using cluster means (μ), not empirical means
    squared_deviations = utils.compute_squared_deviations(
        datapoints, datapoint_indexes, cluster_means, n_clusters
    )

    # Scale squared deviations similar to xy case
    scale = jnp.array([255.0 * 255.0]) / 100.0  # Color range squared / target variance
    squared_deviations = squared_deviations / scale

    # Calculate posterior parameters using conjugate update function
    posterior_alphas, posterior_betas = conjugacy.update_inverse_gamma_normal_conjugacy(
        prior_alphas, prior_betas, squared_deviations, category_counts
    )

    # Sample new sigma values from inverse gamma posterior
    key, subkey = jax.random.split(key)
    new_sigma_rgb = (
        genjax.inverse_gamma.vmap(in_axes=(0, 0))
        .simulate(key, (posterior_alphas, posterior_betas))
        .get_retval()
    )

    # Keep old sigma values for empty clusters
    old_sigma_rgb = trace.get_choices()["blob_model", "sigma_rgb"]
    new_sigma_rgb = jnp.where(
        category_counts[:, None] == 0, old_sigma_rgb, new_sigma_rgb
    )

    # Update trace with new sigma values
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["blob_model", "sigma_rgb"].set(new_sigma_rgb), argdiffs
    )
    return new_trace
