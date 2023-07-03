# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.tree_util as jtu

from genjax._src.core.typing import static_check_is_array


def static_check_broadcast_dim_length(in_axes, args):
    def find_axis_size(axis, x):
        if axis is not None:
            leaves = jtu.tree_leaves(x)
            if leaves:
                return leaves[0].shape[axis]
        return ()

    axis_sizes = jtu.tree_map(find_axis_size, in_axes, args)
    axis_sizes = set(jtu.tree_leaves(axis_sizes))
    if len(axis_sizes) != 1:
        raise ValueError(f"Inconsistent batch axis sizes: {axis_sizes}")
    (d_axis_size,) = axis_sizes
    return d_axis_size


def static_check_leaf_length(tree):
    def _inner(v):
        if static_check_is_array(v):
            shape = v.shape
            return shape[0] if shape else 0
        else:
            return 0

    broadcast_dim_tree = jtu.tree_map(lambda v: _inner(v), tree)
    leaves = jtu.tree_leaves(broadcast_dim_tree)
    leaf_lengths = set(leaves)
    # all the leaves must have the same first dim size.
    assert len(leaf_lengths) == 1
    max_index = list(leaf_lengths).pop()
    return max_index
