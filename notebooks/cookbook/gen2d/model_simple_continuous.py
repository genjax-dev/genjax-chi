"""
This file implements the generative model for the Gen2D project, which extends the Dirichlet mixture model from Gen1D.

The model is designed to work well with block-Gibbs sampling inference, following a co-design approach between model and inference method.

Model Structure:
---------------
1. Global Hyperparameters:
   - sigma_xy: Controls spatial variance
   - sigma_rgb: Controls color variance
   - Additional parameters for inverse gamma distributions

2. Gaussian Mixture Components (N components):
   For each Gaussian i=1..N:
   - Spatial parameters:
     * xy_mean[i] ~ Normal(mu, sigma_xy * I_2)  # 2D spatial mean
     * xy_sigma[i] ~ InverseGamma(a_xy, b_xy)   # Spatial variance

   - Color parameters:
     * rgb_mean[i] ~ Normal(127.5, sigma_rgb)    # Independent RGB means
     * rgb_sigma[i] ~ InverseGamma(a_rgb, b_rgb) # Color variance

   - Mixture weight:
     * weight[i] ~ Gamma(alpha, 1)               # Component weight
     * probs = normalize(weights)                # Normalized mixture probabilities

3. Data Generation (H*W datapoints):
   For each pixel:
   - idx ~ Categorical(probs)                    # Select Gaussian component
   - xy ~ MVNormal(xy_mean[idx], xy_sigma[idx] * I_2)  # Sample spatial location
   - rgb ~ MVNormal(rgb_mean[idx], rgb_sigma[idx] * I_3) # Sample RGB color

The model is implemented using the GenJAX probabilistic programming framework.
"""

import jax.numpy as jnp
from jax._src.basearray import Array

import genjax
from genjax import Pytree, categorical, gamma, gen, inverse_gamma, normal
from genjax._src.core.typing import Array
from genjax.typing import FloatArray


def sample_gamma_safe(key, alpha, beta):
    sample = gamma.sample(key, alpha, beta)
    return jnp.where(sample == 0, 1e-12, sample)


gamma_safe = genjax.exact_density(sample_gamma_safe, gamma.logpdf)

MID_PIXEL_VAL = 255.0 / 2.0
GAMMA_RATE_PARAMETER = 1.0


@Pytree.dataclass
class Hyperparams(Pytree):
    # Most parameters will be inferred via enumerative Gibbs in revised version

    # Hyper params for prior xy mean and spread of clusters
    mu_xy: jnp.ndarray
    sigma_xy: jnp.ndarray

    # Hyper params for prior rgb spread of clusters
    sigma_rgb: jnp.ndarray

    # Hyper params for xy inverse-gamma, roughly xy width of each cluster
    a_xy: jnp.ndarray
    b_xy: jnp.ndarray

    # Hyper params for rgb inverse-gamma, roughly rgb width of each cluster
    a_rgb: jnp.ndarray
    b_rgb: jnp.ndarray

    # Hyper param for mixture weight
    alpha: float

    # number of Gaussians
    n_blobs: int = Pytree.static()

    # Image size
    H: int = Pytree.static()
    W: int = Pytree.static()


@Pytree.dataclass
class LikelihoodParams(Pytree):
    xy_mean: FloatArray
    xy_spread: FloatArray
    rgb_mean: FloatArray
    rgb_spread: FloatArray
    mixture_probs: FloatArray


@gen
def xy_model(
    blob_idx: int,
    a_xy: jnp.ndarray,
    b_xy: jnp.ndarray,
    mu_xy: jnp.ndarray,
    sigma_xy: jnp.ndarray,
):
    xy_spread = inverse_gamma.vmap(in_axes=(0, 0))(a_xy, b_xy) @ "sigma_xy"

    xy_mean = normal.vmap(in_axes=(0, 0))(mu_xy, sigma_xy) @ "xy_mean"
    return xy_mean, xy_spread


@gen
def rgb_model(
    blob_idx: int, a_rgb: jnp.ndarray, b_rgb: jnp.ndarray, sigma_rgb: jnp.ndarray
):
    rgb_spread = inverse_gamma.vmap(in_axes=(0, 0))(a_rgb, b_rgb) @ "sigma_rgb"

    rgb_mean = normal.vmap(in_axes=(None, 0))(MID_PIXEL_VAL, sigma_rgb) @ "rgb_mean"
    return rgb_mean, rgb_spread


@gen
def blob_model(blob_idx: int, hypers: Hyperparams):
    a_xy = hypers.a_xy
    b_xy = hypers.b_xy
    mu_xy = hypers.mu_xy
    a_rgb = hypers.a_rgb
    b_rgb = hypers.b_rgb
    alpha = hypers.alpha
    sigma_xy = hypers.sigma_xy
    sigma_rgb = hypers.sigma_rgb

    xy_mean, xy_spread = xy_model.inline(blob_idx, a_xy, b_xy, mu_xy, sigma_xy)
    rgb_mean, rgb_spread = rgb_model.inline(blob_idx, a_rgb, b_rgb, sigma_rgb)
    mixture_weight = gamma_safe(alpha, GAMMA_RATE_PARAMETER) @ "mixture_weight"
    return xy_mean, xy_spread, rgb_mean, rgb_spread, mixture_weight


@gen
def likelihood_model(
    pixel_idx: int,
    params: LikelihoodParams,
):
    blob_idx: Array = categorical(params.mixture_probs) @ "blob_idx"
    xy_mean: Array = params.xy_mean[blob_idx]
    xy_spread = params.xy_spread[blob_idx]
    rgb_mean = params.rgb_mean[blob_idx]
    rgb_spread = params.rgb_spread[blob_idx]

    xy = normal.vmap(in_axes=(0, 0))(xy_mean, xy_spread) @ "xy"
    rgb = normal.vmap(in_axes=(0, 0))(rgb_mean, rgb_spread) @ "rgb"
    return xy, rgb


@gen
def model(hypers: Hyperparams):
    xy_mean, xy_spread, rgb_mean, rgb_spread, mixture_weights = (
        blob_model.vmap(in_axes=(0, None))(jnp.arange(hypers.n_blobs), hypers)
        @ "blob_model"
    )

    # TODO: should I use them in logspace?
    mixture_probs = mixture_weights / sum(mixture_weights)
    likelihood_params = LikelihoodParams(
        xy_mean, xy_spread, rgb_mean, rgb_spread, mixture_probs
    )
    idxs = jnp.arange(hypers.H * hypers.W)

    _ = (
        likelihood_model.vmap(in_axes=(0, None))(idxs, likelihood_params)
        @ "likelihood_model"
    )

    return None
