import numpy as np
import jax.numpy as jnp
import jax
from fast_3dp3.model import make_scoring_function
from fast_3dp3.rendering import render_planes
from fast_3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from fast_3dp3.shape import get_cube_shape, get_rectangular_prism_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import genjax
import matplotlib.pyplot as plt
import cv2

original_fx, original_fy =  385.798, 385.798
original_cx, original_cy = 321.49, 244.092

cm = plt.get_cmap('turbo')


scaling_factor = 0.25

fx = original_fx * scaling_factor
fy = original_fy * scaling_factor

cx = original_cx * scaling_factor
cy = original_cy * scaling_factor

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
])
original_height = 480
original_width = 640
h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))
print(h,w,fx,fy,cx,cy)

fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx,cy])

shape = get_rectangular_prism_shape(0.13 / 2.0, 0.19 / 2.0, 0.04 / 2.0)

key = jax.random.PRNGKey(3)

key, *sub_keys = jax.random.split(key, 150)
sub_keys = jnp.array(sub_keys)
def f(key):
    key, (_, v) = genjax.VonMisesFisher.random_weighted(
        key, jnp.array([1.0, 0.0, 0.0, 0.0]), 10.0
    )
    r =  quaternion_to_rotation_matrix(v)
    return jnp.vstack(
        [jnp.hstack([r, jnp.array([0.0, 0.0, 1.0]).reshape(3,1) ]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

f_jit = jax.jit(jax.vmap(f))
rotation_deltas = f_jit(sub_keys)

max_depth = 1.0


render_jit = jax.jit(lambda pose, shape: render_planes(pose, shape, h, w, fx_fy, cx_cy))
images = []
for pose in rotation_deltas:
    rendered_image = render_jit(pose, shape)
    images.append(
        Image.fromarray(
            (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), 
            mode="RGBA"
        )
    )

images[0].save(
    fp="out_depth.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)
