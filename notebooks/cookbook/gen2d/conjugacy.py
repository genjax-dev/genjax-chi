"""
This module contains conjugate pair update functions for the Gen2D model, used in Gibbs sampling inference.

The conjugate pairs implemented are:
- Dirichlet-Categorical: For updating cluster weights based on point assignments
- Normal-Normal: For updating cluster means based on assigned points

Each function takes the prior parameters and observations, and returns the posterior parameters
according to the conjugate update equations.
"""

import jax
import jax.numpy as jnp
import model_simple_continuous

from genjax import dirichlet, gamma


def dirichlet_categorical_update(key, associations, n_clusters, alpha):
    """Returns (categorical_vector, metadata_dict)."""

    def get_assoc_count(cluster_idx):
        masked_relevant_datapoint_indices = tiling.relevant_datapoints_for_blob(
            cluster_idx
        )
        relevant_associations = associations[masked_relevant_datapoint_indices.value]
        return jnp.sum(
            jnp.logical_and(
                masked_relevant_datapoint_indices.flag,
                relevant_associations == cluster_idx,
            )
        )

    assoc_counts = jax.vmap(get_assoc_count)(jnp.arange(n_clusters))
    prior_alpha = alpha
    post_alpha = prior_alpha + assoc_counts
    return dirichlet(post_alpha)(key), {}


def conjugate_dirichlet_categorical(
    key, associations, n_clusters, alpha, λ=model_simple_continuous.GAMMA_RATE_PARAMETER
):
    """
    Conjugate update for the case where we have
        X_i ~ Gamma(alpha_i / n, lambda) for i = 1, 2, ..., n;
        X_0 := sum_i X_i
        p := [X_1, X_2, ..., X_n] / X_0
        Y_i ~ Categorical(p) for i = 1, 2, ..., m.

    Here, `n_clusters` is `n`, `associations` is `Y`,
    and `alpha_vec_for_gamma_distributions[i-1]` is `alpha_i`.

    Returns (mixture_weights, metadata), where `mixture_weights`
    is the same thing as the vector `[X_1, X_2, ..., X_n]`.
    """
    ## Derivation of this update:
    # With notation as the above, it turns out
    # X_0 ~ Gamma(alpha.sum(), lambda),
    # p ~ Dirichlet(alpha_1, alpha_2, ..., alpha_n),
    # and X_0 and p are independent.
    # Thus, the posterior on (X_0, p) is
    # p ~ dirichlet_categorical_posterior(alpha, n, assoc_counts);
    # X_0 ~ gamma(alpha.sum(), lambda). # Ie. same as the prior.
    k1, k2 = jax.random.split(key)
    posterior_pvec, _ = dirichlet_categorical_update(
        k1, associations, n_clusters, alpha
    )
    total = gamma(alpha.sum(), λ)(k2)
    return posterior_pvec * total, {}


# Conjugate update for Normal-iid-Normal distribution
def update_normal_normal_conjugacy(
    prior_mean, prior_variance, likelihood_mean, likelihood_variance, category_counts
):
    """Compute posterior parameters for Normal-Normal conjugate update.

    Given a Normal prior N(prior_mean, prior_variance) and Normal likelihood
    N(likelihood_mean, likelihood_variance/n) for n i.i.d. observations,
    computes the parameters of the posterior Normal distribution.

    Args:
        prior_mean: Array of shape (D,) containing prior means
        prior_variance: Array of shape (N,D) containing prior variances
        likelihood_mean: Array of shape (N,D) containing empirical means of observations
        likelihood_variance: Array of shape (D,) containing likelihood variances
        category_counts: Array of shape (N,) containing number of observations per group

    Returns:
        Tuple of:
        - posterior_means: Array of shape (N,D) containing posterior means
        - posterior_variances: Array of shape (N,D) containing posterior variances
    """
    # Expand dimensions to align shapes for broadcasting
    prior_mean = prior_mean[None, :]  # (1,2)
    likelihood_variance = likelihood_variance[None, :]  # (1,2)
    category_counts = category_counts[:, None]  # (10,1)

    posterior_means = (
        prior_variance
        / (prior_variance + likelihood_variance / category_counts)
        * likelihood_mean
        + (likelihood_variance / category_counts)
        / (prior_variance + likelihood_variance / category_counts)
        * prior_mean
    )

    posterior_variances = 1 / (
        1 / prior_variance + category_counts / likelihood_variance
    )

    return posterior_means, posterior_variances
