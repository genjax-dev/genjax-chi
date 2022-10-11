import jax.numpy as jnp


def get_rectangular_prism_shape(s_x,s_y,s_z):
    cube_plane_poses = jnp.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, s_z],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -s_z],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, s_y],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -s_y],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, s_x],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, -s_x],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    plane_dimensions = jnp.array(
        [[s_x, s_y], [s_x, s_y], [s_x, s_z], [s_x, s_z], [s_z, s_y], [s_z, s_y]]
    )
    return cube_plane_poses, plane_dimensions

def get_cube_shape(side_length):
    half_width = side_length / 2.0
    cube_plane_poses = jnp.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, half_width],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -half_width],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, half_width],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -half_width],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, half_width],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, -half_width],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    plane_dimensions = jnp.array(
        [
            [half_width, half_width],
            [half_width, half_width],
            [half_width, half_width],
            [half_width, half_width],
            [half_width, half_width],
            [half_width, half_width],
        ]
    )
    return cube_plane_poses, plane_dimensions
