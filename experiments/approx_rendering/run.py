import jax.numpy as jnp
import jax
from fast_3dp3.model import make_scoring_function
from fast_3dp3.rendering import render_cloud_at_pose
from fast_3dp3.utils import make_centered_grid_enumeration_3d_points, make_cube_point_cloud, quaternion_to_rotation_matrix
import time
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R

import numpy as np
import genjax

h, w, fx_fy, cx_cy = (
    100,
    200,
    jnp.array([200.0, 200.0]),
    jnp.array([100.0, 50.0]),
)
r = 0.1
outlier_prob = 0.01
pixel_smudge = 0

num_frames = 50


gt_poses = [
    jnp.array([
    [1.0, 0.0, 0.0, -1.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 10.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
]

delta_pose = jnp.array([
    [1.0, 0.0, 0.0, 0.05],   
    [0.0, 1.0, 0.0, 0.02],   
    [0.0, 0.0, 1.0, 0.04],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
for t in range(num_frames):
    gt_poses.append(gt_poses[-1].dot(delta_pose))


gt_images = jnp.stack([render_cloud_at_pose(object_model_cloud, p,h,w,fx_fy,cx_cy, pixel_smudge) for p in gt_poses])
print(gt_images.shape)
print((gt_images[0,:,:,-1] > 0 ).sum())


scorer = make_scoring_function(object_model_cloud, h, w, fx_fy, cx_cy ,r, outlier_prob, pixel_smudge)
score = scorer(key, gt_poses[0], gt_images[0,:,:,:])
print(score)

scorer_parallel = jax.vmap(scorer, in_axes = (0, 0, None))



key = jax.random.PRNGKey(3)
key, *sub_keys = jax.random.split(key, 20)
f = jax.jit(lambda k: quaternion_to_rotation_matrix(genjax.VonMisesFisher.sample(k,jnp.array([1.0, 0.0, 0.0, 0.0]),1000.0)))
rotation_deltas = [
    f(sub_key)
    for sub_key in sub_keys
]
grid = make_centered_grid_enumeration_3d_points(0.1, 0.1, 0.1, 3, 3, 3)
pose_deltas = [
    jnp.vstack([jnp.hstack([R, t.reshape(3,1)]), jnp.array([0.0, 0.0, 0.0, 1.0])])
    for R in rotation_deltas for t in grid
]
pose_deltas = jnp.stack(pose_deltas)

object_model_cloud = make_cube_point_cloud(0.5, 8)

print("grid ", pose_deltas.shape)
key, *sub_keys = jax.random.split(key, pose_deltas.shape[0] + 1)
sub_keys = jnp.array(sub_keys)

def _inner(x, gt_image):
    for _ in range(3):
        proposals = jnp.einsum('ij,ajk->aik', x, pose_deltas)
        _, weights_new, x = scorer_parallel(sub_keys, proposals , gt_image)
        x = proposals[jnp.argmax(weights_new)]
    return x, x

def inference(init_pos, gt_images):
    return jax.lax.scan(_inner, init_pos, gt_images)


inference_jit = jax.jit(inference)
a = inference_jit(gt_poses[0], gt_images)

start = time.time()
_, inferred_poses = inference_jit(gt_poses[0], gt_images);
end = time.time()
print ("Time elapsed:", end - start)



def render(object_model_cloud, pose):
    return render_cloud_at_pose(object_model_cloud, pose,h,w,fx_fy,cx_cy, pixel_smudge)
render_cloud_at_pose_jit = jax.jit(render)

for (i,gt_image) in enumerate(gt_images):
    plt.clf()
    plt.imshow(gt_image[:,:,-1])
    plt.savefig("/tmp/{}.png".format(i))

    pose = inferred_poses[i]
    rendered_image = render_cloud_at_pose_jit(object_model_cloud, pose)
    plt.clf()
    plt.imshow(rendered_image[:,:,-1])
    plt.savefig("/tmp/{}_synth.png".format(i))



# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif

imgs = []


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

for (i,gt_image) in enumerate(gt_images):
    im1 = Image.open("/tmp/{}.png".format(i))
    im2 = Image.open("/tmp/{}_synth.png".format(i))
    full_img = get_concat_h(im1, im2)
    imgs.append(full_img)


img = imgs[0]
img.save(fp="out.gif", format='GIF', append_images=imgs,
         save_all=True, duration=20, loop=0)

r, c = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
pixel_coords = jnp.stack([r,c],axis=-1)
print(pixel_coords[20,100,:])
print(pixel_coords[21,100,:])

pixel_coords_dir_xy =  (pixel_coords - cx_cy) / fx_fy
print(pixel_coords_dir_xy[20,100,:])
print(pixel_coords_dir_xy[21,100,:])
print(pixel_coords_dir_xy[50,100,:])

pixel_coords_dir = jnp.concatenate([pixel_coords_dir_xy, jnp.ones((*pixel_coords_dir_xy.shape[:2],1))],axis=-1)

print(pixel_coords[20,50,:])
print(pixel_coords[21,50,:])
print(pixel_coords_dir[20,50,:])
print(pixel_coords_dir[21,50,:])

plane_poses = jnp.array([[
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 1.0, 20.0],   
    [0.0, 0.0, 0.0, 1.0],   
],
[
    [1.0, 0.0, 0.0, 2.0],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 1.0, 20.0],   
    [0.0, 0.0, 0.0, 1.0],   
]])

denom = pixel_coords_dir.dot(plane_pose[:3,2].reshape(3,1))
d = plane_pose[:3,3].reshape(-1,3).dot(plane_pose[:3,2]) / denom
points = pixel_coords_dir * d
points = jnp.concatenate([points, jnp.ones((*points.shape[:2],1))],axis=-1)

print(points[50,50,:])
print(points[51,50,:])

dimensions = jnp.array([0.5, 0.5])

inv_plane_pose = jnp.linalg.inv(plane_pose)
points_in_plane_frame = jnp.einsum("ij,abj", inv_plane_pose, points)

valid = jnp.all(jnp.abs(points_in_plane_frame[:,:,:2]) < dimensions,axis=-1)
print(valid.sum())
points_masked = points * valid[:,:,None]




plt.clf()
plt.imshow(points_masked[:,:,-1])
plt.savefig("out.png")




point_cloud = jnp.vstack([-1.0 * jnp.ones((1, 3)), transformed_cloud])



from IPython import embed; embed()