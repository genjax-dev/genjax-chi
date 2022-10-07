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

# console = genjax.go_pretty()


data = np.load("data.npz")
depth_imgs = np.array(data["depth"]).copy() * 0.001
rgb_imgs = np.array(data["rgb"]).copy() 

original_fx, original_fy =  385.798, 385.798
original_cx, original_cy = 321.49, 244.092

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

coord_images = [
    depth_to_coords_in_camera(
        cv2.resize(
            cv2.bilateralFilter(d.copy().astype(np.float32), 4, 1.0, 1.0),
            (w,h),interpolation=1
        ),
        K.copy()
    )[0]
    for d in depth_imgs
]
gt_images = np.stack(coord_images)
# gt_images[gt_images[:,:,:,2] > 0.8] = 0.0
# gt_images[gt_images[:,:,:,2] < 0.3] = 0.0
# gt_images[gt_images[:,:,:,1] > 0.1,:] = 0.0
gt_images = np.concatenate([gt_images, np.ones(gt_images.shape[:3])[:,:,:,None] ], axis=-1)

cm = plt.get_cmap('turbo')


max_depth = 3.0
images = []
for i in range(gt_images.shape[0]):
    images.append(
        Image.fromarray(
            np.rint(cm(np.array(gt_images[i, :, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
        )
    )

images[0].save(
    fp="depth_out.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)

from IPython import embed; embed()

# images = []
# for rgb in rgb_imgs:
#     images.append(
#         Image.fromarray(
#             rgb.astype(np.int8), mode="RGB"
#         )
#     )

# images[0].save(
#     fp="rgb_out.gif",
#     format="GIF",
#     append_images=images,
#     save_all=True,
#     duration=100,
#     loop=0,
# )


for i in range(3):
    plt.clf()
    plt.imshow(gt_images[-1,:,:,i])
    plt.colorbar()
    plt.savefig("{}.png".format(i))



fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx,cy])
gt_images = jnp.array(gt_images)

r = 0.005
outlier_prob = 0.2
num_frames = 50

shape = get_rectangular_prism_shape(0.10 / 2.0, 0.17 / 2.0, 0.03 / 2.0)

key = jax.random.PRNGKey(4)

first_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.00],
        [0.0, 1.0, 0.0, -0.30],
        [0.0, 0.0, 1.0, 0.65],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy, r, outlier_prob)
score = scorer(key, first_pose, gt_images[0, :, :, :])
scorer_parallel = jax.vmap(scorer, in_axes=(0, 0, None))

key, *sub_keys = jax.random.split(key, 1000)
sub_keys = jnp.array(sub_keys)
def f(key):
    key, (_, v) = genjax.VonMisesFisher.random_weighted(
        key, jnp.array([1.0, 0.0, 0.0, 0.0]), 500.0
    )
    r =  quaternion_to_rotation_matrix(v)
    return jnp.vstack(
        [jnp.hstack([r, jnp.zeros((3, 1)) ]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )
f_jit = jax.jit(jax.vmap(f))
rotation_deltas = f_jit(sub_keys)

f_jit = jax.jit(jax.vmap(lambda t:     jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )))
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.1, 0.1, 0.1, 10, 10, 10))

print("grid ", rotation_deltas.shape)
key, *sub_keys = jax.random.split(key, rotation_deltas.shape[0] + 1)
sub_keys_rotation = jnp.array(sub_keys)


print("grid ", pose_deltas.shape)
key, *sub_keys = jax.random.split(key, pose_deltas.shape[0] + 1)
sub_keys_translation = jnp.array(sub_keys)


def _inner(x, gt_image):
    for _ in range(10):
        proposals = jnp.einsum("ij,ajk->aik", x, pose_deltas)
        _, weights_new, x = scorer_parallel(sub_keys_translation, proposals, gt_image)
        x = proposals[jnp.argmax(weights_new)]

        proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
        _, weights_new, x = scorer_parallel(sub_keys_rotation, proposals, gt_image)
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
cm = plt.get_cmap('turbo')

render_jit = jax.jit(lambda pose, shape: render_planes(pose, shape, h, w, fx_fy, cx_cy))


images = []
for i in range(gt_images.shape[0]):
    pose = inferred_poses[i]
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


# images = []
# for i in range(gt_images.shape[0]):
#     dst = Image.new(
#         "RGBA", (2 * original_width + middle_width, original_height)
#     )

#     rgb = rgb_imgs[i]
#     rgb_img = Image.fromarray(
#         rgb.astype(np.int8), mode="RGB"
#     ).convert("RGBA")

#     dst.paste(
#         rgb_img,
#         (0, 0),
#     )

#     dst.paste(
#         Image.new("RGBA", (middle_width, original_height), (255, 255, 255, 255)),
#         (original_width, 0),
#     )

#     pose = inferred_poses[i]
#     rendered_image = render_jit(pose, shape)
#     overlay_image_1 = Image.fromarray(
#         (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8),
#         mode="RGBA"
#     ).resize((original_width,original_height))
#     overlay_image_1.putalpha(128)
#     rgb_img_copy = rgb_img.copy()
#     rgb_img_copy.putalpha(128)

#     dst.paste(
#         Image.alpha_composite(overlay_image_1, rgb_img_copy),
#         (original_width + middle_width, 0),
#     )
#     images.append(dst)



# images[0].save(
#     fp="out.gif",
#     format="GIF",
#     append_images=images,
#     save_all=True,
#     duration=100,
#     loop=0,
# )



images = []
for i in range(gt_images.shape[0]):
    dst = Image.new(
        "RGBA", (2 * w + middle_width, h)
    )

    # rgb = rgb_imgs[i]
    # rgb_img = Image.fromarray(
    #     rgb.astype(np.int8), mode="RGB"
    # ).convert("RGBA")
    rgb_img = Image.fromarray(
        np.rint(cm(np.array(gt_images[i, :, :, 2]) / max_depth) * 255.0).astype(np.int8),
        mode="RGBA"
    )
    
    dst.paste(
        rgb_img,
        (0, 0),
    )

    dst.paste(
        Image.new("RGBA", (middle_width, h), (255, 255, 255, 255)),
        (w, 0),
    )

    pose = inferred_poses[i]
    rendered_image = render_jit(pose, shape)
    overlay_image_1 = Image.fromarray(
        (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8),
        mode="RGBA"
    ).resize((w,h))
    overlay_image_1.putalpha(128)
    rgb_img_copy = rgb_img.copy()
    rgb_img_copy.putalpha(128)

    dst.paste(
        Image.alpha_composite(overlay_image_1, rgb_img_copy),
        (w + middle_width, 0),
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
