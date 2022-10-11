import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fast_3dp3.model import make_scoring_function
from fast_3dp3.rendering import render_planes
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from fast_3dp3.shape import get_cube_shape
from fast_3dp3.utils import make_centered_grid_enumeration_3d_points
from fast_3dp3.utils import quaternion_to_rotation_matrix
from PIL import Image
from scipy.spatial.transform import Rotation as R

import genjax
import matplotlib.pyplot as plt
import cv2

# console = genjax.go_pretty()


data = np.load("data.npz")
depth_imgs = np.array(data["depth_imgs"]).copy()
rgb_imgs = np.array(data["rgb_imgs"]).copy()

scaling_factor = 0.25

fx = data["fx"] * scaling_factor
fy = data["fy"] * scaling_factor

cx = data["cx"] * scaling_factor
cy = data["cy"] * scaling_factor

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
])
original_height = data["height"]
original_width = data["width"]
h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))
print(h,w,fx,fy,cx,cy)

coord_images = [depth_to_coords_in_camera(cv2.resize(d.copy(), (w,h),interpolation=1), K.copy())[0] for d in depth_imgs]
gt_images = np.stack(coord_images)
gt_images[gt_images[:,:,:,2] > 40.0] = 0.0
gt_images[gt_images[:,:,:,1] > 0.85,:] = 0.0
gt_images = np.concatenate([gt_images, np.ones(gt_images.shape[:3])[:,:,:,None] ], axis=-1)

# images = []
# for i in range(gt_images.shape[0]):
#     images.append(
#         Image.fromarray(
#             np.rint(cm(np.array(gt_images[i, :, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
#         )
#     )
# images[0].save(
#     fp="out.gif",
#     format="GIF",
#     append_images=images,
#     save_all=True,
#     duration=100,
#     loop=0,
# )


fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx,cy])
gt_images = jnp.array(gt_images)

r = 0.1
outlier_prob = 0.1
num_frames = 50

shape = get_cube_shape(2.0)



<<<<<<< HEAD
delta_pose = jnp.array(
=======
key = jax.random.PRNGKey(3)

first_pose = jnp.array(
>>>>>>> save
    [
        [1.0, 0.0, 0.0, -5.00],
        [0.0, 1.0, 0.0, -4.00],
        [0.0, 0.0, 1.0, 20.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
<<<<<<< HEAD
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

=======
>>>>>>> save
scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy, r, outlier_prob)
score = scorer(key, first_pose, gt_images[0, :, :, :])
scorer_parallel = jax.vmap(scorer, in_axes=(0, 0, None))

key, *sub_keys = jax.random.split(key, 150)
sub_keys = jnp.array(sub_keys)


def f(key):
    key, (_, v) = genjax.VonMisesFisher.random_weighted(
        key, jnp.array([1.0, 0.0, 0.0, 0.0]), 700.0
    )
    r = quaternion_to_rotation_matrix(v)
    return jnp.vstack(
        [jnp.hstack([r, jnp.zeros((3, 1))]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )


f_jit = jax.jit(jax.vmap(f))
rotation_deltas = f_jit(sub_keys)

<<<<<<< HEAD

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
=======
f_jit = jax.jit(jax.vmap(lambda t:     jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )))
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.2, 0.2, 0.2, 5, 5, 5))
>>>>>>> save

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
a = inference_jit(first_pose, gt_images)

start = time.time()
_, inferred_poses = inference_jit(first_pose, gt_images)
end = time.time()
elapsed = end - start
print("Time elapsed:", elapsed)
print("FPS:", gt_images.shape[0] / elapsed)

middle_width = 20
<<<<<<< HEAD
cm = plt.get_cmap("turbo")
max_depth = 10.0
=======
cm = plt.get_cmap('turbo')
max_depth = 50.0

render_jit = jax.jit(lambda pose, shape: render_planes(pose, shape, h, w, fx_fy, cx_cy))

>>>>>>> save


images = []
for i in range(gt_images.shape[0]):
    dst = Image.new(
        "RGBA", (2 * original_width + middle_width, original_height)
    )

    rgb = rgb_imgs[i]
    rgb_img = Image.fromarray(
        rgb.astype(np.int8), mode="RGBA"
    )

    dst.paste(
<<<<<<< HEAD
        Image.fromarray(
            np.rint(
                cm(np.array(gt_images[i, :, :, 2]) / max_depth) * 255.0
            ).astype(np.int8),
            mode="RGBA",
        ),
=======
        rgb_img,
>>>>>>> save
        (0, 0),
    )

    dst.paste(
<<<<<<< HEAD
        Image.new(
            "RGBA", (middle_width, gt_images.shape[1]), (255, 255, 255, 255)
        ),
        (gt_images.shape[2], 0),
=======
        Image.new("RGBA", (middle_width, original_height), (255, 255, 255, 255)),
        (original_width, 0),
>>>>>>> save
    )

    pose = inferred_poses[i]
    rendered_image = render_jit(pose, shape)
    overlay_image_1 = Image.fromarray(
        (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    ).resize((original_width,original_height))
    overlay_image_1.putalpha(128)
    rgb_img_copy = rgb_img.copy()
    rgb_img_copy.putalpha(128)

    dst.paste(
<<<<<<< HEAD
        Image.fromarray(
            (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(
                np.int8
            ),
            mode="RGBA",
        ),
        (gt_images.shape[2] + middle_width, 0),
=======
        Image.alpha_composite(overlay_image_1, rgb_img_copy),
        (original_width + middle_width, 0),
>>>>>>> save
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
<<<<<<< HEAD

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
=======
>>>>>>> save
