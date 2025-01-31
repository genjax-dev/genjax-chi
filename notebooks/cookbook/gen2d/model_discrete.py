import jax.numpy as jnp

from genjax import Pytree, categorical, gen, inverse_gamma, normal
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
    sigma_x = inverse_gamma(hypers.a_x, hypers.b_x) @ "sigma_x"
    sigma_y = inverse_gamma(hypers.a_y, hypers.b_y) @ "sigma_y"

    x_mean = normal(hypers.mu_x, sigma_x) @ "x_mean"
    y_mean = normal(hypers.mu_y, sigma_y) @ "y_mean"
    return jnp.array([x_mean, y_mean])


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
    mixture_weight = gamma(hypers.alpha, GAMMA_RATE_PARAMETER) @ "mixture_weight"

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

    xy = normal.vmap(in_axes=(0, 0))(xy_mean, hypers.sigma_xy) @ "xy"
    rgb = normal.vmap(in_axes=(0, 0))(rgb_mean, hypers.sigma_rgb) @ "rgb"
    return xy, rgb
    return None


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
