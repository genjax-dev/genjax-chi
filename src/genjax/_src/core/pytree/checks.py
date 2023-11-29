# Copyright 2023 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax.tree_util as jtu

from genjax._src.core.typing import List
from genjax._src.core.typing import static_check_is_array


def static_check_tree_structure_equivalence(trees: List):
    if not trees:
        return True
    else:
        fst, *rest = trees
        treedef = jtu.tree_structure(fst)
        check = all(map(lambda v: treedef == jtu.tree_structure(v), rest))
        return check


def static_check_tree_leaves_have_matching_leading_dim(tree):
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
