### This file describes the generative model used in the Gen2D project.
#
# This is an extension of the Dirichlet mixture model used in the Gen1D model
# The model is designed so that block-Gibbs is expected to do well for inference on the model, i.e. the model has be co-designed with the inference we
# are running on it.
#
# The generative model is described as follows.
# - Fix hyperparameters sigma_xy, sigma_rgb and other global parameters like the ones for the inverse gamma distributions.
# - Generate N Gaussians
#   For each Gaussian
#       xy_mean ~ Normal(mu, sigma_xy * eye(2))
#       xy_sigma ~ InverseGamma(…some parameter…)
#       r_mean, g_mean, b_mean {~}_{iid} Normal(127.5, sigma_rgb)
#       rgb_sigma ~ InverseGamma(…some parameter…)
#  - mixture_weight ~ Gamma(alpha, 1) # alpha is an arg to the model
#    probs = normalize([gaussian.mixture_weight for gaussian in gaussians])
#  - Generate a H*W array of datapoints.
#   For each datapoint:
#       idx ~ categorical([probs])
#       mygaussian = gaussians[idx]
#       xy ~ mvnormal(mygaussian.xy_mean, mygaussian.sigma_xy * eye(2))
#       rgb ~ mvnormal(mygaussian.rgb_mean, mygaussian.sigma_rgb * eye(3))


import jax.numpy as jnp
from jax._src.basearray import Array

import genjax
from genjax import Pytree, categorical, gamma, gen, inverse_gamma, normal
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

    # Hyper params for xy inverse-gamma
    a_xy: jnp.ndarray
    b_xy: jnp.ndarray

    # Hyper params for prior mean on xy
    mu_xy: jnp.ndarray

    # Hyper params for rgb inverse-gamma
    a_rgb: jnp.ndarray
    b_rgb: jnp.ndarray

    # Hyper param for mixture weight
    alpha: float

    # Hyper params for noise in likelihood
    sigma_xy: jnp.ndarray
    sigma_rgb: jnp.ndarray

    # number of Gaussians
    n_blobs: int = Pytree.static()

    # Image size
    H: int = Pytree.static()
    W: int = Pytree.static()


@gen
def xy_model(blob_idx: int, hypers: Hyperparams):
    sigma_xy = inverse_gamma.vmap(in_axes=(0, 0))(hypers.a_xy, hypers.b_xy) @ "sigma_xy"

    xy_mean = normal.vmap(in_axes=(0, 0))(hypers.mu_xy, sigma_xy) @ "xy_mean"
    return xy_mean


@gen
def rgb_model(blob_idx: int, hypers: Hyperparams):
    rgb_sigma = (
        inverse_gamma.vmap(in_axes=(0, 0))(hypers.a_rgb, hypers.b_rgb) @ "sigma_rgb"
    )

    rgb_mean = normal.vmap(in_axes=(None, 0))(MID_PIXEL_VAL, rgb_sigma) @ "rgb_mean"
    return rgb_mean


@gen
def blob_model(blob_idx: int, hypers: Hyperparams):
    xy_mean = xy_model.inline(blob_idx, hypers)
    rgb_mean = rgb_model.inline(blob_idx, hypers)
    mixture_weight = gamma_safe(hypers.alpha, GAMMA_RATE_PARAMETER) @ "mixture_weight"
    return xy_mean, rgb_mean, mixture_weight


@Pytree.dataclass
class LikelihoodParams(Pytree):
    xy_mean: FloatArray
    rgb_mean: FloatArray
    mixture_probs: FloatArray


@gen
def likelihood_model(pixel_idx: int, params: LikelihoodParams, hypers: Hyperparams):
    blob_idx = categorical(params.mixture_probs) @ "blob_idx"
    xy_mean: Array = params.xy_mean[blob_idx]
    rgb_mean = params.rgb_mean[blob_idx]

    xy = normal.vmap(in_axes=(0, 0))(xy_mean, hypers.sigma_xy) @ "xy"
    rgb = normal.vmap(in_axes=(0, 0))(rgb_mean, hypers.sigma_rgb) @ "rgb"
    return xy, rgb


@gen
def model(hypers: Hyperparams):
    xy_mean, rgb_mean, mixture_weights = (
        blob_model.vmap(in_axes=(0, None))(jnp.arange(hypers.n_blobs), hypers)
        @ "blob_model"
    )

    # TODO: should I use them in logspace?
    mixture_probs = mixture_weights / sum(mixture_weights)
    likelihood_params = LikelihoodParams(xy_mean, rgb_mean, mixture_probs)

    _ = (
        likelihood_model.vmap(in_axes=(0, None, None))(
            jnp.arange(hypers.H * hypers.W), likelihood_params, hypers
        )
        @ "likelihood_model"
    )

    return None
