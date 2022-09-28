import numpy as np
import jax.numpy as jnp
import jax
from fast_3dp3.model import make_scoring_function
from fast_3dp3.rendering import render_cloud_at_pose
from fast_3dp3.utils import make_centered_grid_enumeration_3d_points
import time
import matplotlib.pyplot as plt
from PIL import Image

object_model_cloud = jnp.array(np.random.rand(200, 3) * 2.0)
h, w, fx_fy, cx_cy = (
    120,
    160,
    jnp.array([200.0, 200.0]),
    jnp.array([60.0, 80.0]),
)
r = 0.2
outlier_prob = 0.01

num_frames = 100
gt_poses = [
    jnp.array([
        [1.0, 0.0, 0.0, x],   
        [0.0, 1.0, 0.0, y],   
        [0.0, 0.0, 1.0, 20.0],   
        [0.0, 0.0, 0.0, 1.0],   
        ]
    )
    for (x,y) in np.hstack([np.linspace(-3,3,num_frames).reshape(-1,1),np.linspace(-3,3,num_frames).reshape(-1,1)])
]
gt_images = jnp.stack([render_cloud_at_pose(object_model_cloud, p,h,w,fx_fy,cx_cy) for p in gt_poses])
print(gt_images.shape)
print((gt_images[0,:,:,-1] > 0 ).sum())

key = jax.random.PRNGKey(3)
scorer = make_scoring_function(object_model_cloud, h, w, fx_fy, cx_cy ,r, outlier_prob)
score = scorer(key, jnp.zeros(3), gt_images[0,:,:,:])
print(score)

scorer_parallel = jax.vmap(scorer, in_axes = (0, 0, None))

grid = make_centered_grid_enumeration_3d_points(0.1, 0.1, 0.1, 10, 10, 10)
print("grid ", grid.shape)
key, *sub_keys = jax.random.split(key, grid.shape[0] + 1)
sub_keys = jnp.array(sub_keys)


def _inner(x, gt_image):
    proposals = grid + x
    _, weights_new, x = scorer_parallel(sub_keys, proposals , gt_image)
    x = proposals[jnp.argmax(weights_new)]
    return x, x

def inference(init_pos, gt_images):
    return jax.lax.scan(_inner, init_pos, gt_images)


inference_jit = jax.jit(inference)

a = inference_jit(gt_poses[0][:3,3], gt_images)

start = time.time()
_, inferred_poses = inference_jit(gt_poses[0][:3,3], gt_images);
end = time.time()
print ("Time elapsed:", end - start)



def render(object_model_cloud, pose):
    return render_cloud_at_pose(object_model_cloud, pose,h,w,fx_fy,cx_cy)
render_cloud_at_pose_jit = jax.jit(render)

for (i,gt_image) in enumerate(gt_images):
    plt.clf()
    plt.imshow(gt_image[:,:,-1])
    plt.savefig("/tmp/{}.png".format(i))

    pos = inferred_poses[i]
    pose = jnp.array([
        [1.0, 0.0, 0.0, pos[0]],   
        [0.0, 1.0, 0.0, pos[1]],   
        [0.0, 0.0, 1.0, pos[2]],   
        [0.0, 0.0, 0.0, 1.0],   
        ]
    )
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
