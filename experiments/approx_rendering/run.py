import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fast_3dp3.model import make_scoring_function
from fast_3dp3.rendering import render_planes
from fast_3dp3.shape import get_cube_shape
from fast_3dp3.utils import make_centered_grid_enumeration_3d_points
from fast_3dp3.utils import quaternion_to_rotation_matrix
from PIL import Image
from scipy.spatial.transform import Rotation as R

import genjax


console = genjax.go_pretty()

h, w, fx_fy, cx_cy = (
    150,
    150,
    jnp.array([200.0, 200.0]),
    jnp.array([80.0, 60.0]),
)
r = 0.1
outlier_prob = 0.01
num_frames = 50

shape = get_cube_shape(1.0)


delta_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.05],
        [0.0, 1.0, 0.0, 0.08],
        [0.0, 0.0, 1.0, 0.19],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
delta_pose = delta_pose.at[:3, :3].set(
    jnp.array(R.from_euler("xy", [np.pi / 100, np.pi / 20]).as_matrix())
)
gt_poses = [
    jnp.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
]
for t in range(num_frames):
    gt_poses.append(gt_poses[-1].dot(delta_pose))

render_jit = jax.jit(
    lambda pose, shape: render_planes(pose, shape, h, w, fx_fy, cx_cy)
)
gt_images = jnp.stack([render_jit(p, shape) for p in gt_poses])

key = jax.random.PRNGKey(3)

scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy, r, outlier_prob)
score = scorer(key, gt_poses[0], gt_images[0, :, :, :])
scorer_parallel = jax.vmap(scorer, in_axes=(0, 0, None))

key, *sub_keys = jax.random.split(key, 100)
sub_keys = jnp.array(sub_keys)


def f(key):
    key, (_, v) = genjax.VonMisesFisher.random_weighted(
        key, jnp.array([1.0, 0.0, 0.0, 0.0]), 1000.0
    )
    r = quaternion_to_rotation_matrix(v)
    return jnp.vstack(
        [jnp.hstack([r, jnp.zeros((3, 1))]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )


f_jit = jax.jit(jax.vmap(f))
rotation_deltas = f_jit(sub_keys)


f_jit = jax.jit(
    jax.vmap(
        lambda t: jnp.vstack(
            [
                jnp.hstack([jnp.eye(3), t.reshape(3, -1)]),
                jnp.array([0.0, 0.0, 0.0, 1.0]),
            ]
        )
    )
)
pose_deltas = f_jit(
    make_centered_grid_enumeration_3d_points(0.2, 0.2, 0.2, 5, 5, 5)
)


print("grid ", rotation_deltas.shape)
key, *sub_keys = jax.random.split(key, rotation_deltas.shape[0] + 1)
sub_keys_rotation = jnp.array(sub_keys)


print("grid ", pose_deltas.shape)
key, *sub_keys = jax.random.split(key, pose_deltas.shape[0] + 1)
sub_keys_translation = jnp.array(sub_keys)


def _inner(x, gt_image):
    for _ in range(3):
        proposals = jnp.einsum("ij,ajk->aik", x, pose_deltas)
        _, weights_new, x = scorer_parallel(
            sub_keys_translation, proposals, gt_image
        )
        x = proposals[jnp.argmax(weights_new)]

        proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
        _, weights_new, x = scorer_parallel(
            sub_keys_rotation, proposals, gt_image
        )
        x = proposals[jnp.argmax(weights_new)]

    return x, x


def inference(init_pos, gt_images):
    return jax.lax.scan(_inner, init_pos, gt_images)


inference_jit = jax.jit(inference)
a = inference_jit(gt_poses[0], gt_images)

start = time.time()
_, inferred_poses = inference_jit(gt_poses[0], gt_images)
end = time.time()
elapsed = end - start
print("Time elapsed:", elapsed)
print("FPS:", gt_images.shape[0] / elapsed)

middle_width = 20
cm = plt.get_cmap("turbo")
max_depth = 10.0


images = []
for i in range(gt_images.shape[0]):
    dst = Image.new(
        "RGBA", (2 * gt_images.shape[2] + middle_width, gt_images.shape[1])
    )
    dst.paste(
        Image.fromarray(
            np.rint(
                cm(np.array(gt_images[i, :, :, 2]) / max_depth) * 255.0
            ).astype(np.int8),
            mode="RGBA",
        ),
        (0, 0),
    )

    dst.paste(
        Image.new(
            "RGBA", (middle_width, gt_images.shape[1]), (255, 255, 255, 255)
        ),
        (gt_images.shape[2], 0),
    )

    pose = inferred_poses[i]
    rendered_image = render_jit(pose, shape)
    dst.paste(
        Image.fromarray(
            (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(
                np.int8
            ),
            mode="RGBA",
        ),
        (gt_images.shape[2] + middle_width, 0),
    )
    images.append(dst)


images[0].save(
    fp="out.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)

from IPython import embed


embed()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


data = np.load("data.npz")
depth = data["depth"]
print(depth.shape)
fx = data["fx"]
cx = data["cx"]
fy = data["fy"]
cy = data["cy"]
h = data["height"]
w = data["width"]
print(h, w, fx, fy, cx, cy)

cm = plt.get_cmap("turbo")

Image.fromarray(
    np.rint(cm(np.array(depth) / 30.0) * 255.0).astype(np.int8), mode="RGBA"
).save("out2.png")

shape = get_cube_shape(2.0)
pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.00],
        [0.0, 1.0, 0.0, -4.00],
        [0.0, 0.0, 1.0, 20.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
img = render_planes(
    pose, shape, h, w, jnp.array([fx, fy]), jnp.array([cx, cy])
)

Image.fromarray(
    np.rint(cm(np.array(img[:, :, 2]) / 30.0) * 255.0).astype(np.int8),
    mode="RGBA",
).save("out3.png")
