"""Utility functions for the Gen2D model.

This module provides utility functions for working with the Gen2D model, including:

- Pytree indexing utilities
- Trace manipulation functions for extracting and merging parameters
- Conjugacy update helper functions for Gibbs sampling
"""

import jax
import jax.numpy as jnp
import gen2d.model_simple_continuous as model_simple_continuous

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Pytree, Trace
from genjax._src.core.generative.choice_map import ChoiceMap


### Pytree indexing utils ###
def mywhere(b, x, y):
    """
    Like jnp.where(b, x, y), but can handle cases like
    b.shape = (N,) while x.shape = y.shape = (N, M, ...).
    """
    assert len(x.shape) == len(y.shape)
    if len(b.shape) == len(x.shape):
        return jnp.where(b, x, y)
    else:
        return jnp.where(
            b[:, *(None for _ in range(len(x.shape) - len(b.shape)))], x, y
        )


### Manipulating traces ###
def markov_for_xy_mean_from_trace(tracediff):
    """Extract XY mean-related parameters from trace for Gibbs update.

    This function extracts all parameters needed for updating XY means via Gibbs sampling:
    - Cluster assignments for each datapoint
    - XY coordinates for each datapoint
    - Number of clusters
    - Prior mean location
    - Current XY means for each cluster
    - Current XY variances for each cluster
    - Observation variance

    Args:
        trace: GenJAX trace containing current model state

    Returns:
        Tuple containing:
        - datapoint_indexes: Array of cluster assignments for each point
        - datapoints: Array of XY coordinates for each point
        - n_clusters: Integer number of clusters
        - prior_mean: Array of prior XY mean values
        - cluster_xy_means: Array of current XY means per cluster
        - cluster_xy_variances: Array of current XY variances per cluster
        - obs_variance: Observation variance parameter
    """
    datapoint_indexes = tracediff.chm["likelihood_model", "blob_idx"]
    datapoints = tracediff.chm["likelihood_model", "xy"]
    n_clusters = tracediff.args[0].n_blobs
    prior_mean = tracediff.args[0].mu_xy
    cluster_xy_means = tracediff.chm["blob_model", "xy_mean"]  # shape (N,2)
    cluster_xy_variances = tracediff.chm["blob_model", "sigma_xy"]
    obs_variance = tracediff.args[0].sigma_xy

    return (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        cluster_xy_means,
        cluster_xy_variances,
        obs_variance,
    )


def markov_for_rgb_mean_from_trace(tracediff):
    """Extract RGB mean-related parameters from trace for Gibbs update.

    This function extracts all parameters needed for updating RGB means via Gibbs sampling:
    - Cluster assignments for each datapoint
    - RGB values for each datapoint
    - Number of clusters
    - Prior mean (mid pixel value)
    - Current RGB means for each cluster
    - Current RGB variances for each cluster
    - Observation variance

    Args:
        trace: GenJAX trace containing current model state

    Returns:
        Tuple containing:
        - datapoint_indexes: Array of cluster assignments for each point
        - datapoints: Array of RGB values for each point
        - n_clusters: Integer number of clusters
        - prior_mean: Array of prior RGB mean values
        - cluster_rgb_means: Array of current RGB means per cluster
        - cluster_rgb_variances: Array of current RGB variances per cluster
        - obs_variance: Observation variance parameter
    """
    datapoint_indexes = tracediff.chm["likelihood_model", "blob_idx"]
    datapoints = tracediff.chm["likelihood_model", "rgb"]
    n_clusters = tracediff.args[0].n_blobs
    prior_mean = model_simple_continuous.MID_PIXEL_VAL * jnp.ones(3)
    cluster_rgb_means = tracediff.chm["blob_model", "rgb_mean"]  # shape (N,3)
    cluster_rgb_variances = tracediff.chm["blob_model", "sigma_rgb"]
    obs_variance = tracediff.args[0].sigma_rgb

    return (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        cluster_rgb_means,
        cluster_rgb_variances,
        obs_variance,
    )


def extract_blob_params(trace):
    """Extract blob parameters from a model trace.

    Extracts the spatial variances (sigma_xy), RGB variances (sigma_rgb),
    spatial means (xy_mean), RGB means (rgb_mean), and mixture weights
    from the blob model choices in the trace.

    Args:
        trace: A model execution trace containing blob model parameters

    Returns:
        tuple: A 5-tuple containing:
            - sigma_xy: Spatial variance parameters for each cluster
            - sigma_rgb: RGB variance parameters for each cluster
            - xy_mean: Spatial mean parameters for each cluster
            - rgb_mean: RGB mean parameters for each cluster
            - mixture_weight: Mixture weights for each cluster
    """
    sigma_xy = trace.get_choices()["blob_model", "sigma_xy"]
    sigma_rgb = trace.get_choices()["blob_model", "sigma_rgb"]
    xy_mean = trace.get_choices()["blob_model", "xy_mean"]
    rgb_mean = trace.get_choices()["blob_model", "rgb_mean"]
    mixture_weight = trace.get_choices()["blob_model", "mixture_weight"]
    return sigma_xy, sigma_rgb, xy_mean, rgb_mean, mixture_weight


def create_blob_observations(sigma_xy, sigma_rgb, xy_mean, rgb_mean, mixture_weight):
    """Create an observation object from blob parameters.

    Args:
        sigma_xy: Spatial variance parameters for each cluster
        sigma_rgb: RGB variance parameters for each cluster
        xy_mean: Spatial mean parameters for each cluster
        rgb_mean: RGB mean parameters for each cluster
        mixture_weight: Mixture weights for each cluster

    Returns:
        obs: Combined observation object with all blob parameters
    """
    obs = (
        C["blob_model", "sigma_xy"].set(sigma_xy)
        | C["blob_model", "sigma_rgb"].set(sigma_rgb)
        | C["blob_model", "xy_mean"].set(xy_mean)
        | C["blob_model", "rgb_mean"].set(rgb_mean)
        | C["blob_model", "mixture_weight"].set(mixture_weight)
    )
    return obs


def extract_likelihood(trace):
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    datapoints_xy = trace.get_choices()["likelihood_model", "xy"]
    datapoints_rgb = trace.get_choices()["likelihood_model", "rgb"]
    return (
        C["likelihood_model", "blob_idx"].set(datapoint_indexes)
        | C["likelihood_model", "xy"].set(datapoints_xy)
        | C["likelihood_model", "rgb"].set(datapoints_rgb)
    )


@Pytree.dataclass
class TraceDiff(Pytree):
    """A class representing a compact trace representation containing arguments and choice map.

    This class is used instead of full traces during inference.
    It contains the original arguments and a choice map
    representing the changes/choices made.

    Args:
        args: Tuple of arguments from the original trace
        chm: ChoiceMap containing the choices/changes made in this trace difference
    """

    args: tuple
    chm: ChoiceMap


def merge(key, trace: Trace, tracediff: TraceDiff):
    """Merge a TraceDiff back into a full Trace.

    Args:
        key: JAX random key
        trace: Original trace to merge into
        tracediff: TraceDiff containing changes to merge

    Returns:
        new_trace: Updated trace with changes from tracediff merged in
    """
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(key, tracediff.chm, argdiffs)
    return new_trace


def extract(trace) -> TraceDiff:
    """Extract a TraceDiff from a full Trace. Specialized for the Gen2D model.

    Creates a TraceDiff containing the blob parameters and likelihood choices
    from the given trace.

    Args:
        trace: Trace to extract from

    Returns:
        TraceDiff containing the extracted parameters and choices
    """
    sigma_xy, sigma_rgb, xy_mean, rgb_mean, mixture_weight = extract_blob_params(trace)
    chm1 = create_blob_observations(
        sigma_xy, sigma_rgb, xy_mean, rgb_mean, mixture_weight
    )
    chm2 = extract_likelihood(trace)
    return TraceDiff(args=trace.args, chm=chm1 | chm2)


def concat(tracediff: TraceDiff, chm) -> TraceDiff:
    """Concatenate a ChoiceMap with a TraceDiff's existing choices.
     The choices form the ChoiceMap overwrite existing ones from the TraceDiff.

    Args:
        tracediff: Original TraceDiff
        chm: ChoiceMap to concatenate with tracediff's choices

    Returns:
        New TraceDiff with concatenated choices
    """
    new_chm = chm | tracediff.chm
    return TraceDiff(args=tracediff.args, chm=new_chm)


### Conjugacy updates helpers ###
def category_count(datapoint_indexes, n_clusters):
    """Count the number of points assigned to each cluster.

    Args:
        datapoint_indexes: Array of shape (N,) containing cluster assignments for each point
        n_clusters: Integer number of clusters to count

    Returns:
        Array of shape (n_clusters,) containing count of points per cluster
    """
    return jnp.bincount(
        datapoint_indexes,
        length=n_clusters,
        minlength=n_clusters,
    )


def compute_means(datapoints, datapoint_indexes, n_clusters, category_counts):
    """Compute the mean of datapoints for each cluster.

    Args:
        datapoints: Array of shape (N, D)
        datapoint_indexes: Array of shape (N,) containing cluster assignments
        n_clusters: Integer number of clusters
        category_counts: Array of shape (n_clusters,) containing counts per cluster

    Returns:
        Array of shape (n_clusters, D) containing mean coordinates per cluster
    """
    # Use segment_sum for more efficient summation
    sums = jax.ops.segment_sum(datapoints, datapoint_indexes, n_clusters)
    safe_counts = jnp.maximum(category_counts, 1)
    means = sums / safe_counts[:, None]
    return means


def compute_squared_deviations(datapoints, cluster_indices, means, n_clusters):
    """Compute sum of squared deviations from cluster means.

    Args:
        datapoints: Array of shape (N, D) containing observations
        cluster_indices: Array of shape (N,) containing cluster assignments
        means: Array of shape (K, D) containing cluster means
        n_clusters: Number of clusters K

    Returns:
        Array of shape (K, D) containing sum of squared deviations per cluster
    """

    def sum_squared_devs(cluster_idx):
        # Compute (x - μ)² for all points
        diffs = datapoints - means[cluster_idx]
        squared_diffs = diffs**2

        # Use where to mask points not in this cluster
        masked_diffs = jnp.where(
            (cluster_indices == cluster_idx)[:, None], squared_diffs, 0.0
        )

        # Sum over all points
        return jnp.sum(masked_diffs, axis=0)

    # Compute for each cluster
    deviations = jax.vmap(sum_squared_devs)(jnp.arange(n_clusters))
    return deviations
