import jax.numpy as jnp
import functools
from .utils import apply_transform, extract_2d_patches

# @functools.partial(jax.jit, static_argnames=["r", "outlier_prob"])
def neural_descriptor_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob
):
    obs_mask = obs_xyz[:,:,2] > 0
    rendered_mask = rendered_xyz[:,:,2] > 0
    num_latent_points = rendered_mask.sum()
    rendered_xyz_patches = extract_2d_patches(rendered_xyz, (4,4))
    log_mixture_prob = log_likelihood_for_pixel(
        obs_xyz,
        rendered_xyz_patches,
        r,
        outlier_prob,
        num_latent_points
    )
    return jnp.sum(jnp.where(obs_mask, log_mixture_prob, 0.0))


@functools.partial(
    jnp.vectorize,
    signature='(m),(h,w,m)->()',
    excluded=(2, 3, 4),
)
def log_likelihood_for_pixel(
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    r: float,
    outlier_prob: float,
    num_latent_points: float,
):
    """    Args:
        data_xyz (jnp.ndarray): (3,)
            3d coordinate of observed point
        model_xyz : (filter_height, filter_width, 3),
        r : float, sphere radius
        outlier_prob: float
        num_latent_points: int
    """
    distance = jnp.linalg.norm(data_xyz - model_xyz, axis=-1).ravel() # (4,4)
    
    return jnp.log(jnp.sum(outlier_prob + jnp.where(
        distance <= r,
        1 / (4 * ((1 - outlier_prob)/num_latent_points) * jnp.pi * r**3) / (3 ),
        0.0,
    )))
    return a
