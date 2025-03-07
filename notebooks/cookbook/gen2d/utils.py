from dataclasses import replace
from typing import TypeVar

import jax
import jax.numpy as jnp
import model_simple_continuous

import genjax
from genjax import Const, PythonicPytree

T = TypeVar("T")

### Taken from gen3d/src/gen3d/variants/condor/utils.py


def sample_dirichlet_safe(key, alpha):
    if alpha.shape == (1,):
        return jnp.array([1.0])
    sample = genjax.dirichlet.sample(key, alpha)
    return jnp.where(sample == 0, 1e-12, sample)


def logpdf_dirichlet_safe(val, alpha):
    if alpha.shape == (1,):
        return jnp.array([0.0])
    return genjax.dirichlet.logpdf(val, alpha)


dirichlet = genjax.exact_density(sample_dirichlet_safe, logpdf_dirichlet_safe)


def unwrap(x):
    """Unwrap `x` if it is a `Const`; otherwise return `x`."""
    if isinstance(x, Const):
        return x.val
    else:
        return x


class MyPytree(PythonicPytree):
    """
    Pytree base class with some extra bells and whistles, including:
        - supports self.replace(...) to functionally update fields
    """

    def replace(self: T, **kwargs) -> T:
        return replace(self, **kwargs)

    @staticmethod
    def eq(x, y):
        # See https://github.com/probcomp/genjax/issues/1441 for why
        # I didn't just override __eq__.
        # (Could get the __eq__ override to work with a bit more effort, however.)
        if jax.tree_util.tree_structure(x) != jax.tree_util.tree_structure(y):
            return False
        leaves1 = jax.tree_util.tree_leaves(x)
        leaves2 = jax.tree_util.tree_leaves(y)
        bools = [jnp.all(l1 == l2) for l1, l2 in zip(leaves1, leaves2)]
        return jnp.all(jnp.array(bools))

    # The __getitem__ override is needed for GenJAX versions
    # prior to https://github.com/probcomp/genjax/pull/1440.
    def __getitem__(self, idx):
        return jax.tree_util.tree_map(lambda v: v[idx], self)


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


def markov_for_xy_mean_from_trace(trace):
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
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    datapoints = trace.get_choices()["likelihood_model", "xy"]
    n_clusters = trace.args[0].n_blobs
    prior_mean = trace.args[0].mu_xy
    cluster_xy_means = trace.get_choices()["blob_model", "xy_mean"]  # shape (N,2)
    cluster_xy_variances = trace.get_choices()["blob_model", "sigma_xy"]
    obs_variance = trace.args[0].sigma_xy

    return (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        cluster_xy_means,
        cluster_xy_variances,
        obs_variance,
    )


def markov_for_rgb_mean_from_trace(trace):
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
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    datapoints = trace.get_choices()["likelihood_model", "rgb"]
    n_clusters = trace.args[0].n_blobs
    prior_mean = model_simple_continuous.MID_PIXEL_VAL * jnp.ones(3)
    cluster_rgb_means = trace.get_choices()["blob_model", "rgb_mean"]  # shape (N,3)
    cluster_rgb_variances = trace.get_choices()["blob_model", "sigma_rgb"]
    obs_variance = trace.args[0].sigma_rgb

    return (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        cluster_rgb_means,
        cluster_rgb_variances,
        obs_variance,
    )


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
