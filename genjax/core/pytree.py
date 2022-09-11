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

"""Contains the Pytree class."""

import abc
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from genjax.core.specialization import is_concrete

__all__ = [
    "Pytree",
]


class Pytree(metaclass=abc.ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_node(
            cls,
            cls.flatten,
            cls.unflatten,
        )

    @abc.abstractmethod
    def flatten(self):
        pass

    @classmethod
    @abc.abstractmethod
    def unflatten(cls, data, xs):
        pass

    # This is overloaded here to support generic interfaces
    # to `genjax.Selection.filter`.
    def get_score(self):
        return 0.0


#####
# Pytree sum type
#####

# If you have multiple Pytrees, you might want
# to generate a "sum" Pytree with leaves that minimally cover
# the entire set.


#####
# Utilities
#####


def tree_stack(trees):
    """
    Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).

    Useful for turning a list of objects into something you can feed to a
    vmapped function.

    This function respects concrete vs. traced values. It will leave concrete
    leaves unchanged (it will not lift them to :code:`jax.core.Tracer`).
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jtu.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [
        np.stack(leaf) if all(map(is_concrete, leaf)) else jnp.stack(leaf)
        for leaf in grouped_leaves
    ]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """
    Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]

    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jtu.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees


def squeeze(tree):
    def _inner(v):
        if isinstance(v, np.ndarray):
            return np.squeeze(v)
        elif isinstance(v, jnp.ndarray):
            return jnp.squeeze(v)
        else:
            return v

    return jtu.tree_map(_inner, tree)
