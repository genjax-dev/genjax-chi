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

def render_planes(pose, shape, h,w, fx_fy, cx_cy):
    shape_planes, shape_dimensions = shape
    plane_poses = jnp.einsum("ij,ajk",pose, shape_planes)

    r, c = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
    pixel_coords = jnp.stack([r,c],axis=-1)
    pixel_coords_dir = jnp.concatenate([(pixel_coords - cx_cy) / fx_fy, jnp.ones((h,w,1))],axis=-1)

    denoms = jnp.einsum("ijk,ak->ija", pixel_coords_dir , plane_poses[:,:3,2])
    numerators = jnp.einsum("...k,...k", 
        plane_poses[:,:3,3],
        plane_poses[:,:3,2]
    )
    d = numerators / denoms
    points_temp = jnp.einsum("...ij,...kj", d[:,:,:,None], pixel_coords_dir[:,:,:,None])
    points = jnp.concatenate([points_temp, jnp.ones((*points_temp.shape[:3],1,))],axis=-1) # (H,W,N,4)
    inv_plane_poses = jnp.linalg.inv(plane_poses)
    points_in_plane_frame = jnp.einsum("...ij,ab...j->ab...i", inv_plane_poses, points)

    valid = jnp.all(jnp.abs(points_in_plane_frame[:,:,:,:2]) < shape_dimensions,axis=-1) # (H,W,N)
    z_vals = (1000.0 - points[:,:,:,2]) * valid
    idxs = jnp.argmax(z_vals, axis=-1)

    points_final = (
        points[jnp.arange(points.shape[0])[:, None], jnp.arange(points.shape[1])[None, :], idxs]
        *
        jnp.any(valid, axis=-1)[:,:,None]
    )

    return points_final
