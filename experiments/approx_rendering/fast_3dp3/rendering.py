import jax.numpy as jnp
from .utils import apply_transform

# render single object model cloud at specified pose to a "coordinate image"
# "coordinate image" is (h,w,3) where at each pixel we have the 3d coordinate of that point
# input_cloud: (N,3) object model point cloud
# pose: (4,4) pose matrix. rotation matrix in top left (3,3) and translation in (:3,3)
# h,w : height and width
# fx_fy : focal lengths
# cx_cy : principal point
# output: (h,w,3) coordinate image
# @functools.partial(jax.jit, static_argnames=["h","w"])
def render_cloud_at_pose(input_cloud, pose, h, w, fx_fy, cx_cy, pixel_smudge):
    transformed_cloud = apply_transform(input_cloud, pose)
    point_cloud = jnp.vstack([-1.0 * jnp.ones((1, 3)), transformed_cloud])

    point_cloud_normalized = point_cloud / point_cloud[:, 2].reshape(-1, 1)
    temp1 = point_cloud_normalized[:, :2] * fx_fy
    temp2 = temp1 + cx_cy
    pixels = jnp.round(temp2)

    x, y = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
    matches = (jnp.abs(x[:, :, None] - pixels[:, 0]) <= pixel_smudge) & (jnp.abs(y[:, :, None] - pixels[:, 1]) <= pixel_smudge)
    matches = matches * (1000.0 - point_cloud[:,-1][None, None, :])

    a = jnp.argmax(matches, axis=-1)    
    return point_cloud[a]
