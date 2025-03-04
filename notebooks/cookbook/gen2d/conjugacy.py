### This file contains different function for conjugate pairs updates for the Gen2D model, which are used in the Gibbs-inference loop.


# def conjugate_update_mvnormal_with_known_cov(
#     prior_mean,  # (D,)
#     prior_cov,  # (D, D)
#     obs_cov,  # (D, D)
#     obs,  # (M, D)
# ):
#     """
#     Returns the posterior mean and covariance for the mean
#     of a multivariate normal distribution with known covariance.
#     That is, given
#       mu ~ Normal(prior_mean, prior_cov),
#       obs_i ~ Normal(mu, obs_cov) for i = 0, 1, ..., M-1,
#     this function returns (post_mean, post_cov) where
#       P(mu | obs) = Normal(post_mean, post_cov).
#     """
#     M = obs.shape[0]
#     post_cov = jnp.linalg.inv(jnp.linalg.inv(prior_cov) + M * jnp.linalg.inv(obs_cov))
#     obsmean = jnp.sum(obs) / M
#     post_mean = post_cov @ (
#         jnp.linalg.inv(prior_cov) @ prior_mean + M * jnp.linalg.inv(obs_cov) @ obsmean
#     )
#     return jnp.where(M > 0, post_mean, prior_mean), jnp.where(
#         M > 0, post_cov, prior_cov
#     )


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
# TODO: currently busted because tensor shape
def update_normal_normal_conjugacy(
    prior_mean, prior_variance, likelihood_mean, likelihood_variance, category_counts
):
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
