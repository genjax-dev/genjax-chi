import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fast_3dp3.rendering import render_planes
from scipy.spatial.transform import Rotation as R


h, w, fx_fy, cx_cy = (
    100,
    200,
    jnp.array([200.0, 200.0]),
    jnp.array([100.0, 50.0]),
)
r = 0.1
outlier_prob = 0.01


cube_plane_poses = jnp.array(
    [
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.0, 0.0, 1.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ]
)

plane_dimensions = jnp.array(
    [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
)

cube_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

cube_pose = cube_pose.at[:3, :3].set(
    jnp.array(R.from_euler("xy", [np.pi / 4, np.pi / 4]).as_matrix())
)


shape = (cube_plane_poses, plane_dimensions)


def render_planes_lambda(pose, shape):
    return render_planes(pose, shape, h, w, fx_fy, cx_cy)


output = render_planes_lambda(cube_pose, shape)
plt.clf()
plt.imshow(output[:, :, 2])
plt.colorbar()
plt.savefig("out.png")

render_planes_lambda_parallel = jax.vmap(
    render_planes_lambda,
    in_axes=(
        0,
        None,
    ),
)

cube_poses_many = jnp.stack([cube_pose for _ in range(2000)])

f = jax.jit(render_planes_lambda_parallel)
output = f(cube_poses_many, shape)


start = time.time()
output = f(cube_poses_many, shape)
end = time.time()
print("Time elapsed:", end - start)


plt.clf()
plt.imshow(output[0, :, :, 2])
plt.colorbar()
plt.savefig("out.png")
