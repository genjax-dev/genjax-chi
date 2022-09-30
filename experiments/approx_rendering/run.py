import numpy as np
import jax.numpy as jnp
import jax
from fast_3dp3.model import make_scoring_function
from fast_3dp3.rendering import render_planes
from fast_3dp3.utils import make_centered_grid_enumeration_3d_points, make_cube_point_cloud, quaternion_to_rotation_matrix
import time
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R
import genjax

h, w, fx_fy, cx_cy = (
    120,
    160,
    jnp.array([200.0, 200.0]),
    jnp.array([80.0, 60.0]),
)
r = 0.1
outlier_prob = 0.01

num_frames = 50


cube_plane_poses = jnp.array([[
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 1.0, 0.5],   
    [0.0, 0.0, 0.0, 1.0],   
],[
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 1.0, -0.5],   
    [0.0, 0.0, 0.0, 1.0],   
],[
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 0.0, -1.0, 0.5],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 0.0, 1.0],   
],[
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 0.0, -1.0, -0.5],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 0.0, 1.0],   
],[
    [0.0, 0.0, 1.0, 0.5],   
    [0.0, 1.0, 0.0, 0.0],   
    [-1.0, 0.0, 0.0, 0.0],   
    [0.0, 0.0, 0.0, 1.0],   
],[
    [0.0, 0.0, 1.0, -0.5],   
    [0.0, 1.0, 0.0, 0.0],   
    [-1.0, 0.0, 0.0, 0.0],   
    [0.0, 0.0, 0.0, 1.0],   
]
])

plane_dimensions = jnp.array([
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5]
]
)
shape = (cube_plane_poses, plane_dimensions)


gt_poses = [
    jnp.array([
    [1.0, 0.0, 0.0, -1.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 5.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
]

delta_pose = jnp.array([
    [1.0, 0.0, 0.0, 0.05],   
    [0.0, 1.0, 0.0, 0.08],   
    [0.0, 0.0, 1.0, 0.19],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
delta_pose = delta_pose.at[:3,:3].set(jnp.array(R.from_euler("xy",[np.pi/100, np.pi/50]).as_matrix()))


for t in range(num_frames):
    gt_poses.append(gt_poses[-1].dot(delta_pose))

def render_planes_lambda(pose, shape):
    return render_planes(pose, shape, h,w, fx_fy, cx_cy)
render_planes_lambda_jit = jax.jit(render_planes_lambda)

key = jax.random.PRNGKey(3)

gt_images = jnp.stack([render_planes_lambda_jit(p, shape) for p in gt_poses])
print(gt_images.shape)
print((gt_images[0,:,:,-1] > 0 ).sum())

scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy ,r, outlier_prob)
score = scorer(key, gt_poses[0], gt_images[0,:,:,:])

scorer_parallel = jax.vmap(scorer, in_axes = (0, 0, None))

key, *sub_keys = jax.random.split(key, 15)
f = jax.jit(lambda k: quaternion_to_rotation_matrix(genjax.VonMisesFisher.sample(k,jnp.array([1.0, 0.0, 0.0, 0.0]),1000.0)))
rotation_deltas = [
    f(sub_key)
    for sub_key in sub_keys
]
grid = make_centered_grid_enumeration_3d_points(0.2, 0.2, 0.2, 4, 4, 4)
pose_deltas = [
    jnp.vstack([jnp.hstack([R, t.reshape(3,1)]), jnp.array([0.0, 0.0, 0.0, 1.0])])
    for R in rotation_deltas for t in grid
]

pose_deltas = jnp.stack(pose_deltas)


print("grid ", pose_deltas.shape)
key, *sub_keys = jax.random.split(key, pose_deltas.shape[0] + 1)
sub_keys = jnp.array(sub_keys)

def _inner(x, gt_image):
    for _ in range(1):
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
elapsed = end - start
print ("Time elapsed:", elapsed)
print ("FPS:", gt_images.shape[0] / elapsed)

images = []
middle_width = 20
for i in range(gt_images.shape[0]):
    dst = Image.new('RGB', (2*gt_images.shape[2] + middle_width, gt_images.shape[1]))
    dst.paste(Image.fromarray(np.array(gt_images[i,:,:,2]) / 10.0 * 255.0, mode="F"),
        (0,0))
        
    dst.paste(Image.new("RGB", (middle_width, gt_images.shape[1]), (255, 255, 255)), (gt_images.shape[2],0))

    pose = inferred_poses[i]
    rendered_image = render_planes_lambda_jit(pose, shape)
    dst.paste(Image.fromarray(np.array(rendered_image[:,:,2]) / 10.0 * 255.0, mode="F"),
        (gt_images.shape[2] + middle_width ,0))
    images.append(dst)

images[0].save(fp="out.gif", format='GIF', append_images=images,
         save_all=True, duration=100, loop=0)

from IPython import embed; embed()