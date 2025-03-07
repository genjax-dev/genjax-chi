"""
This module contains conjugate pair update functions for the Gen2D model, used in Gibbs sampling inference.

The conjugate pairs implemented are:
- Normal-Normal: For updating cluster means based on assigned points
- InverseGamma-Normal: For updating cluster variances based on assigned points

Each function takes the prior parameters and observations, and returns the posterior parameters
according to the conjugate update equations.

The Normal-Normal conjugacy is used to update the means of both the spatial (xy) and color (rgb)
components of each Gaussian cluster. The InverseGamma-Normal conjugacy is used to update the
variances of both components.

The update equations follow standard Bayesian conjugate prior formulas, with careful handling of
vectorized operations across multiple clusters and dimensions.
"""


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


# Conjugate update for InverseGamma-Normal distribution
def update_inverse_gamma_normal_conjugacy(
    prior_alpha, prior_beta, likelihood_mean, category_counts
):
    """Compute posterior parameters for InverseGamma-Normal conjugate update.

    Given an InverseGamma prior IG(alpha, beta) and Normal likelihood
    N(mu, sigma^2) where sigma^2 ~ IG(alpha, beta), computes the parameters
    of the posterior InverseGamma distribution.

    Args:
        prior_alpha: Array of shape (D,) containing prior shape parameters
        prior_beta: Array of shape (D,) containing prior scale parameters
        likelihood_mean: Array of shape (N,D) containing empirical means of observations
        category_counts: Array of shape (N,) containing number of observations per group

    Returns:
        Tuple of:
        - posterior_alpha: Array of shape (N,D) containing posterior shape parameters
        - posterior_beta: Array of shape (N,D) containing posterior scale parameters
    """
    # Expand dimensions to align shapes for broadcasting
    prior_alpha = prior_alpha[None, :]  # (1,D)
    prior_beta = prior_beta[None, :]  # (1,D)
    category_counts = category_counts[:, None]  # (N,1)

    # Update shape parameter
    posterior_alpha = prior_alpha + category_counts / 2

    # Update scale parameter
    # Sum of squared deviations term
    squared_deviations = category_counts * likelihood_mean**2 / 2

    posterior_beta = prior_beta + squared_deviations

    return posterior_alpha, posterior_beta
