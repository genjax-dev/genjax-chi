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

import jax.numpy as jnp
import model_simple_continuous

import genjax
from genjax import ChoiceMapBuilder as C

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


# def dirichlet_categorical_update(key, associations, n_clusters, alpha):
#     """Returns (categorical_vector, metadata_dict)."""

#     def get_assoc_count(cluster_idx):
#         masked_relevant_datapoint_indices = tiling.relevant_datapoints_for_blob(
#             cluster_idx
#         )
#         relevant_associations = associations[masked_relevant_datapoint_indices.value]
#         return jnp.sum(
#             jnp.logical_and(
#                 masked_relevant_datapoint_indices.flag,
#                 relevant_associations == cluster_idx,
#             )
#         )

#     assoc_counts = jax.vmap(get_assoc_count)(jnp.arange(n_clusters))
#     prior_alpha = alpha
#     post_alpha = prior_alpha + assoc_counts
#     return dirichlet(post_alpha)(key), {}


# def conjugate_dirichlet_categorical(
#     key, associations, n_clusters, alpha, λ=model_simple_continuous.GAMMA_RATE_PARAMETER
# ):
#     """
#     Conjugate update for the case where we have
#         X_i ~ Gamma(alpha_i / n, lambda) for i = 1, 2, ..., n;
#         X_0 := sum_i X_i
#         p := [X_1, X_2, ..., X_n] / X_0
#         Y_i ~ Categorical(p) for i = 1, 2, ..., m.

#     Here, `n_clusters` is `n`, `associations` is `Y`,
#     and `alpha_vec_for_gamma_distributions[i-1]` is `alpha_i`.

#     Returns (mixture_weights, metadata), where `mixture_weights`
#     is the same thing as the vector `[X_1, X_2, ..., X_n]`.
#     """
#     ## Derivation of this update:
#     # With notation as the above, it turns out
#     # X_0 ~ Gamma(alpha.sum(), lambda),
#     # p ~ Dirichlet(alpha_1, alpha_2, ..., alpha_n),
#     # and X_0 and p are independent.
#     # Thus, the posterior on (X_0, p) is
#     # p ~ dirichlet_categorical_posterior(alpha, n, assoc_counts);
#     # X_0 ~ gamma(alpha.sum(), lambda). # Ie. same as the prior.
#     k1, k2 = jax.random.split(key)
#     posterior_pvec, _ = dirichlet_categorical_update(
#         k1, associations, n_clusters, alpha
#     )
#     total = gamma(alpha.sum(), λ)(k2)
#     return posterior_pvec * total, {}


# # one option in the mean time is to replace inverse_gamma in the model by a categorical with 64 values.
# def conjugate_update_mean_normal_inverse_gamma():
#     return None


# Extract relevant info for the update from the trace
def markov_for_xy_mean_from_trace(trace):
    datapoint_indexes = trace.get_choices()["likelihood_model", "blob_idx"]
    datapoints = trace.get_choices()["likelihood_model", "xy"]
    n_clusters = trace.get_args()[0].n_blobs
    prior_mean = trace.get_args()[0].mu_xy
    current_means = trace.get_choices()["blob_model", "xy_mean"]  # shape (N,2)
    prior_variance = trace.get_choices()["blob_model", "sigma_xy"]
    obs_variance = trace.get_args()[0].sigma_xy

    return (
        datapoint_indexes,
        datapoints,
        n_clusters,
        prior_mean,
        current_means,
        prior_variance,
        obs_variance,
    )


# Count number of points per cluster
def category_count(datapoint_indexes, n_clusters):
    return jnp.bincount(
        datapoint_indexes,
        length=n_clusters,
        minlength=n_clusters,
    )


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


# Update the trace with new xy_mean
def update_trace_with_xy_mean(key, trace, new_means):
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        key, C["blob_model", "xy_mean"].set(new_means), argdiffs
    )
    return new_trace


def update_xy_mean(key, trace):
    # datapoint_indexes, datapoints, n_clusters, prior_mean, current_means, prior_variance, obs_variance = markov_for_xy_mean_from_trace(trace)

    # category_counts = category_count(datapoint_indexes, n_clusters)

    # cluster_means = compute_means(datapoints, datapoint_indexes, n_clusters, category_counts)

    # posterior_means, posterior_variances = update_normal_normal_conjugacy(prior_mean, prior_variance, cluster_means, obs_variance, category_counts)

    # key, subkey = jax.random.split(key)
    # new_means = xy_mean_resampling(key, hypers, current_means, category_counts, prior_variance, category_counts)

    # new_trace = update_trace_with_xy_mean(subkey, trace, new_means)

    # return new_trace
    return trace


def update_xy_sigma(key, trace):
    # datapoint_indexes, datapoints, n_clusters, prior_mean, current_means, prior_variance, obs_variance = markov_for_xy_mean_from_trace(trace)

    # prior_alphas =
    # prior_betas =
    # posterior_alphas = prior_alphas + n_clusters / 2

    return trace


def update_rgb_mean(key, trace):
    return trace


def update_rgb_sigma(key, trace):
    return trace


def update_cluster_assignment(key, trace):
    return trace


def update_mixture_weight(key, trace):
    return trace
