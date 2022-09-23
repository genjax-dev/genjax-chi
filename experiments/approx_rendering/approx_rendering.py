# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import genjax
import functools
from dataclasses import dataclass
from typing import Tuple

#####
# Rendering likelihood
#####


def apply_transform(coords, transform):
    coords = jnp.einsum(
        "ij,...j->...i",
        transform,
        jnp.concatenate([coords, jnp.ones(coords.shape[:-1] + (1,))], axis=-1),
    )[..., :-1]
    return coords


@functools.partial(jax.jit, static_argnames=["h", "w"])
def render_cloud_at_pose(input_cloud, pose, h, w, fx_fy, cx_cy):
    transformed_cloud = apply_transform(input_cloud, pose)
    point_cloud = jnp.vstack([jnp.zeros((1, 3)), transformed_cloud])

    point_cloud_normalized = point_cloud / point_cloud[:, 2].reshape(-1, 1)
    temp1 = point_cloud_normalized[:, :2] * fx_fy.transpose()
    temp2 = temp1 + cx_cy
    pixels = jnp.round(temp2)

    x, y = jnp.meshgrid(jnp.arange(h), jnp.arange(w))
    matches = (x[:, :, None] == pixels[:, 0]) & (y[:, :, None] == pixels[:, 1])
    a = jnp.argmax(matches, axis=-1)
    return point_cloud[a]


@functools.partial(jax.jit, static_argnames="filter_shape")
def extract_2d_patches(
    data: jnp.ndarray, filter_shape: Tuple[int, int]
) -> jnp.ndarray:
    """For each pixel, extract 2D patches centered at that pixel.
    Args:
        data (jnp.ndarray): Array of shape (H, W, ...)
            data needs to be 2, 3, or 4 dimensional.
        filter_shape (Tuple[int, int]): Size of the patches in H, W dimensions
    Returns:
        extracted_patches: Array of shape (H, W, filter_shape[0], filter_shape[1], C)
            extracted_patches[i, j] == data[
                i - filter_shape[0] // 2:i + filter_shape[0] - filter_shape[0] // 2,
                j - filter_shape[1] // 2:j + filter_shape[1] - filter_shape[1] // 2,
            ]
    """
    assert len(filter_shape) == 2
    output_shape = data.shape + filter_shape
    if data.ndim == 2:
        data = data[..., None, None]
    elif data.ndim == 3:
        data = data[:, :, None]

    padding = [
        (filter_shape[ii] // 2, filter_shape[ii] - filter_shape[ii] // 2 - 1)
        for ii in range(len(filter_shape))
    ]
    extracted_patches = jnp.moveaxis(
        jax.lax.conv_general_dilated_patches(
            lhs=data,
            filter_shape=filter_shape,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=("HWNC", "OIHW", "HWNC"),
        ).reshape(output_shape),
        (-2, -1),
        (2, 3),
    )
    return extracted_patches


def neural_descriptor_likelihood(
    obs_xyz, rendered_xyz: jnp.ndarray, r, outlier_prob
):
    obs_mask = obs_xyz[:, :, -1]
    rendered_mask = rendered_xyz[:, :, -1]
    num_latent_points = rendered_mask.sum()
    rendered_xyz_patches = extract_2d_patches(rendered_xyz, (4, 4))
    log_mixture_prob = log_likelihood_for_pixel(
        obs_xyz, rendered_xyz_patches, r, outlier_prob, num_latent_points
    )
    return jnp.sum(jnp.where(obs_mask, log_mixture_prob, 0.0))


@functools.partial(
    jnp.vectorize,
    signature="(m),(h,w,m)->()",
    excluded=(2, 3, 4),
)
def log_likelihood_for_pixel(
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    r: float,
    p_background: float,
    p_foreground: float,
):
    distance = jnp.linalg.norm(data_xyz - model_xyz, axis=-1).ravel()  # (4,4)
    a = jnp.sum(
        p_background
        + jnp.where(
            distance <= 0.1,
            3 * p_foreground / (4 * jnp.pi * r**3),
            0.0,
        )
    )
    return a


@dataclass
class _NeuralDescriptorLikelihood(genjax.Distribution):
    def sample(self, key, *args, **kwargs):
        return key, ()

    def logpdf(self, key, image, *args):
        return neural_descriptor_likelihood(image, *args)


NeuralDescriptorLikelihood = _NeuralDescriptorLikelihood()

#####
# Model
#####

h, w, fx_fy, cx_cy = (
    120,
    160,
    jnp.array([200.0, 200.0]),
    jnp.array([60.0, 80.0]),
)
r = 0.05
outlier_prob = 0.01


@genjax.gen
def model(key, object_model_cloud):
    key, x = genjax.trace("x", genjax.Uniform)(key, (-5.0, 5.0))
    pose = jnp.array(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rendered_image = render_cloud_at_pose(
        object_model_cloud, pose, h, w, fx_fy, cx_cy
    )
    return genjax.trace("rendered", NeuralDescriptorLikelihood)(
        key, (rendered_image, r, outlier_prob)
    )


#####
# Benchmarks
#####

object_model_cloud = np.random.rand(88, 3)
object_model_cloud = jnp.array(object_model_cloud)
gt_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
gt_image = render_cloud_at_pose(object_model_cloud, gt_pose, h, w, fx_fy, cx_cy)


def evaluate_likelihood(
    key,
):
    x = jax.random.uniform(key, minval=-5.0, maxval=5.0)
    score = jax.scipy.stats.uniform.logpdf(x, -5.0, 5.0)
    latent_pose = jnp.array(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rendered_image = render_cloud_at_pose(
        object_model_cloud, latent_pose, h, w, fx_fy, cx_cy
    )

    ### Make distribution whose logscore is
    score += neural_descriptor_likelihood(
        rendered_image, gt_image, r, outlier_prob
    )
    return score


def test_likelihood_evaluation(benchmark):
    key = jax.random.PRNGKey(3)
    key, *sub_keys = jax.random.split(key, 100 + 1)
    sub_keys = jnp.array(sub_keys)
    vmapped = jax.jit(jax.vmap(evaluate_likelihood, in_axes=0))
    benchmark(vmapped, sub_keys)


def test_importance(benchmark):
    gt_pose = jnp.array(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    gt_image = render_cloud_at_pose(
        object_model_cloud, gt_pose, h, w, fx_fy, cx_cy
    )
    chm = genjax.ChoiceMap.new({("rendered",): gt_image})
    key = jax.random.PRNGKey(3)
    key, *sub_keys = jax.random.split(key, 100 + 1)
    sub_keys = jnp.array(sub_keys)
    key, (_, tr) = benchmark(
        jax.jit(jax.vmap(model.importance, in_axes=(0, None, None))),
        sub_keys,
        chm,
        (object_model_cloud,),
    )
