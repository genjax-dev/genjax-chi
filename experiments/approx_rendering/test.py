import numpy as np
import jax.numpy as jnp
import jax
from fast_3dp3.model import make_scoring_function
from fast_3dp3.rendering import render_cloud_at_pose

object_model_cloud = jnp.array(np.random.rand(200,3) * 2.0)
h,w,fx_fy,cx_cy = 120, 160, jnp.array([200.0, 200.0]), jnp.array([60.0, 80.0])
r = 0.2
outlier_prob =  0.01

gt_pose = jnp.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 20.0],
    [0.0, 0.0, 0.0, 1.0],
])
gt_image = render_cloud_at_pose(object_model_cloud, gt_pose, h,w,fx_fy,cx_cy)
print((gt_image[:,:,2] > 0).sum())


key = jax.random.PRNGKey(3)
scorer = make_scoring_function(object_model_cloud, h, w, fx_fy, cx_cy ,r, outlier_prob)
# score = scorer(key, jnp.zeros(3), gt_image)
# print(score)



# gridding = jnp.linspace(-1.0, 1.0, 5)
# deltas = jnp.stack(jnp.meshgrid(gridding,gridding,gridding),axis=-1)
# deltas = deltas.reshape(-1,3)
# print(deltas.shape)
# key, *sub_keys = jax.random.split(key, len(deltas) + 1)
# sub_keys = jnp.array(sub_keys)

# def update_x(x, gt_image):
#     proposals = x + deltas
#     _, weights_new, x = scorer_parallel_jit(sub_keys, proposals , gt_image);
#     best_idx = jnp.argmax(weights_new)
#     y = proposals[best_idx,:]
#     return y