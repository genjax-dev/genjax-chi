import jax.numpy as jnp
import numpy as np
from typing import Tuple
import jax

def extract_2d_patches(data: jnp.ndarray, filter_shape: Tuple[int, int]) -> jnp.ndarray:
    """For each pixel, extract 2D patches centered at that pixel.
    Args:
        data (jnp.ndarray): Array of shape (H, W, ...)
            data needs to be 2, 3, or 4 dimensional.
        filter_shape (Tuple[int, int]): Size of the patches in H, W dimensions
    Returns:
        extracted_patches: Array of shape (H, W, filter_shape[0], filter_shape[1], C)
            extracted_patches[i, j] == data[
                i - filter_shape[0] // 2:i + filter_shape[0] - filter_shape[0] // 2,
                j - filter_shape[1] // 2:j + filter_shape[1] - filter_shape[1] // 2,
            ]
    """
    assert len(filter_shape) == 2
    output_shape = data.shape + filter_shape
    if data.ndim == 2:
        data = data[..., None, None]
    elif data.ndim == 3:
        data = data[:, :, None]

    padding = [
        (filter_shape[ii] // 2, filter_shape[ii] - filter_shape[ii] // 2 - 1)
        for ii in range(len(filter_shape))
    ]
    extracted_patches = jnp.moveaxis(
        jax.lax.conv_general_dilated_patches(
            lhs=data,
            filter_shape=filter_shape,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=("HWNC", "OIHW", "HWNC"),
        ).reshape(output_shape),
        (-2, -1),
        (2, 3),
    )
    return extracted_patches


# move point cloud to a specified pose
# coords: (N,3) point cloud
# pose: (4,4) pose matrix. rotation matrix in top left (3,3) and translation in (:3,3)
def apply_transform(coords, transform):
    coords = jnp.einsum(
        'ij,...j->...i',
        transform,
        jnp.concatenate([coords, jnp.ones(coords.shape[:-1] + (1,))], axis=-1),
    )[..., :-1]
    return coords

def make_centered_grid_enumeration_3d_points(x,y,z,num_x,num_y,num_z):
    gridding = jnp.linspace(-1.0, 1.0, 5)
    deltas = jnp.stack(jnp.meshgrid(
        jnp.linspace(-x,x,num_x),
        jnp.linspace(-y,y,num_y),
        jnp.linspace(-z,z,num_z)
    ),
        axis=-1)
    deltas = deltas.reshape(-1,3)
    return deltas

def make_cube_point_cloud(side_width, num_points):
    side_half_width = side_width / 2.0
    single_side = np.stack(np.meshgrid(
        np.linspace(-side_half_width, side_half_width, num_points),
        np.linspace(-side_half_width, side_half_width, num_points),
        np.linspace(0.0, 0.0, num_points)
    ),
        axis=-1
    ).reshape(-1,3)

    all_faces = []
    for a in [0,1,2]:
        for side in [-1.0, 1.0]:        
            perm = np.arange(3)
            perm[a] = 2
            perm[2] = a
            face = single_side[:,perm]
            face[:,a] = side * side_half_width
            all_faces.append(face)
    object_model_cloud = np.vstack(all_faces)
    return jnp.array(object_model_cloud)


def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = jnp.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix