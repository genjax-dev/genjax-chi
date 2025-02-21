### INITIAL PLAN:
### exact inference on discretized model
### exact Gibbs move on discretized model
### exact Gibbs for continuous model
### test exact Gibbs
### update for cont model

### What do I want?
# Ideally, model.exact_infer(key, obs, args)
# as well as the generality for sub_model.exact_infer(key, obs, args)
# something cool would be an automatic plotting of the BayesNet of the model.

### Math for complexity of exact inference on this model:
# model:
# P(latents | image, hypers)
# = P(latents, image | hypers) / P(image)
# = P(xy_mean, rgb_mean, mixture_weight | hypers)
# * P(blob_idx | mixture_weight)
# * P(xy | xy_mean rgb, blob_idx, hypers)
# * P( rgb | rgb_mean rgb, blob_idx, hypers)  / P(image) = sum_{xy_mean, rgb_mean, mixture_weight, blob_idx} P(latents, image | hypers)
# size of the sum: |xy_mean| * | rgb_mean| * |mixture_weight| * |blob_idx| * |n_blobs | * | image |
# = 64**2 * 64 ** 3 * 64 * 100 * 10 ** 6
# ~ 7. 10^18, without counting the cost of each eval nor ranging over a set of hyper parameters.
# L4 GPU can do 30 * 10^12 flop per sec -> problem.


### However, that's the naive version not tacking the Markov blanket and conditional independence into account. E.g. in HMM using dynamic programming
# we can reduce the complexity from exponential in the length of the chain to linear. The argument there is as follows for a chain of length 3.
# P(x1, x2, x2 | y1, y2, y3)
# = P(x1 | y1) . P(x2 | x1, y2). P(x3 | x2, y3)
# = (P(x1 , y1) / \sum_{x1} P(x1, y1))
# . (P(x1, x2, y2) / \sum_{x1, x2} P(x1, x2, y2))
# . (...)
# =

# NOTES:
# - I may want to keep the distribution for observed data continuous to simplify and avoid unnecessary 0 likelihood.
# - can compress the image to make inference faster, and one can even do SMC from low res to high res.


import jax.numpy as jnp
from discrete_distributions import (
    discrete_gamma,
    discrete_inverse_gamma,
    discrete_normal,
)

from genjax import Pytree, categorical, gen
from genjax.typing import FloatArray

MID_PIXEL_VAL = 255.0 / 2.0
GAMMA_RATE_PARAMETER = 1.0
HYPER_GRID_SIZE = 64
LATENT_GRID_SIZE = 64


@Pytree.dataclass
class Hyperparams(Pytree):
    # Most parameters will be inferred via enumerative Gibbs in revised version

    # Hyper params for xy inverse-gamma
    a_x: float
    b_x: float
    a_y: float
    b_y: float

    # Hyper params for prior mean on xy
    mu_x: float
    mu_y: float

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
    sigma_x = discrete_inverse_gamma(hypers.a_x, hypers.b_x) @ "sigma_x"
    sigma_y = discrete_inverse_gamma(hypers.a_y, hypers.b_y) @ "sigma_y"

    x_mean = discrete_normal(hypers.mu_x, sigma_x) @ "x_mean"
    y_mean = discrete_normal(hypers.mu_y, sigma_y) @ "y_mean"
    return jnp.array([x_mean, y_mean])


@gen
def rgb_model(blob_idx: int, hypers: Hyperparams):
    rgb_sigma = (
        discrete_inverse_gamma.vmap(in_axes=(0, 0))(hypers.a_rgb, hypers.b_rgb)
        @ "sigma_rgb"
    )

    rgb_mean = (
        discrete_normal.vmap(in_axes=(None, 0))(MID_PIXEL_VAL, rgb_sigma) @ "rgb_mean"
    )
    return rgb_mean


@gen
def blob_model(blob_idx: int, hypers: Hyperparams):
    xy_mean = xy_model.inline(blob_idx, hypers)
    rgb_mean = rgb_model.inline(blob_idx, hypers)
    mixture_weight = (
        discrete_gamma(hypers.alpha, GAMMA_RATE_PARAMETER) @ "mixture_weight"
    )

    return xy_mean, rgb_mean, mixture_weight


@Pytree.dataclass
class LikelihoodParams(Pytree):
    xy_mean: FloatArray
    rgb_mean: FloatArray
    mixture_probs: FloatArray


@gen
def likelihood_model(pixel_idx: int, params: LikelihoodParams, hypers: Hyperparams):
    blob_idx = categorical(params.mixture_probs) @ "blob_idx"
    xy_mean = params.xy_mean[blob_idx]
    rgb_mean = params.rgb_mean[blob_idx]

    xy = discrete_normal.vmap(in_axes=(0, 0))(xy_mean, hypers.sigma_xy) @ "xy"
    rgb = discrete_normal.vmap(in_axes=(0, 0))(rgb_mean, hypers.sigma_rgb) @ "rgb"
    return xy, rgb


@gen
def model(hypers: Hyperparams):
    xy_mean, rgb_mean, mixture_weights = (
        blob_model.vmap(in_axes=(0, None))(jnp.arange(hypers.n_blobs), hypers)
        @ "blob_model"
    )

    mixture_probs = mixture_weights / sum(mixture_weights)
    likelihood_params = LikelihoodParams(xy_mean, rgb_mean, mixture_probs)

    _ = (
        likelihood_model.vmap(in_axes=(0, None, None))(
            jnp.arange(hypers.H * hypers.W), likelihood_params, hypers
        )
        @ "likelihood_model"
    )

    return None
