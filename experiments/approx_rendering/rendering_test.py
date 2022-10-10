import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fast_3dp3.rendering import render_cloud_at_pose
from PIL import Image
from scipy.spatial.transform import Rotation as R


# object_model_cloud = make_centered_grid_enumeration_3d_points(2.0, 2.0, 2.0, 30, 30, 30)


jnp.array(np.random.rand(200, 3) * 2.0)
h, w, fx_fy, cx_cy = (
    120,
    160,
    jnp.array([200.0, 200.0]),
    jnp.array([60.0, 80.0]),
)
r = 0.2
outlier_prob = 0.01
pixel_smudge = 3

num_frames = 100
gt_poses = [
    jnp.array(
        [
            [1.0, 0.0, 0.0, -3.0],
            [0.0, 1.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 20.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
]

rot = R.from_euler("zyx", [1.0, -0.1, -2.0], degrees=True).as_matrix()
delta_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, -0.02],
        [0.0, 1.0, 0.0, -0.02],
        [0.0, 0.0, 1.0, 0.01],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
delta_pose = delta_pose.at[:3, :3].set(jnp.array(rot))

for t in range(num_frames):
    gt_poses.append(gt_poses[-1].dot(delta_pose))

gt_images = jnp.stack(
    [
        render_cloud_at_pose(
            object_model_cloud, p, h, w, fx_fy, cx_cy, pixel_smudge
        )
        for p in gt_poses
    ]
)


for (i, gt_image) in enumerate(gt_images):
    plt.clf()
    plt.imshow(gt_image[:, :, -1])
    plt.savefig(f"/tmp/{i}.png")


imgs = []
for (i, gt_image) in enumerate(gt_images):
    im1 = Image.open(f"/tmp/{i}.png")
    imgs.append(im1)


img = imgs[0]
img.save(
    fp="out.gif",
    format="GIF",
    append_images=imgs,
    save_all=True,
    duration=20,
    loop=0,
)
