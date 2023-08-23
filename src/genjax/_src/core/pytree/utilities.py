# Copyright 2022 MIT Probabilistic Computing Project
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


import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.typing import static_check_supports_grad


#####
# Utilities
#####


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).

    Useful for turning a list of objects into something you can feed to
    a vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jtu.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaf) for leaf in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have
    first dimension k, will make k trees [((a[0], b[0]), c[0]), ...,
    ((a[k], b[k]), c[k])]

    Useful for turning the output of a vmapped function into normal
    objects.
    """
    leaves, treedef = jtu.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees


def tree_grad_split(tree):
    def _grad_filter(v):
        if static_check_supports_grad(v):
            return v
        else:
            return None

    def _nograd_filter(v):
        if not static_check_supports_grad(v):
            return v
        else:
            return None

    grad = jtu.tree_map(_grad_filter, tree)
    nograd = jtu.tree_map(_nograd_filter, tree)

    return grad, nograd


def tree_zipper(grad, nograd):
    def _zipper(*args):
        for arg in args:
            if arg is not None:
                return arg
        return None

    def _is_none(x):
        return x is None

    return jtu.tree_map(_zipper, grad, nograd, is_leaf=_is_none)
