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
"""
This module contains an abstract data class (called `Pytree`) for implementing JAX's [`Pytree` interface](https://jax.readthedocs.io/en/latest/pytrees.html) on derived classes.

The Pytree interface determines how data classes behave across JAX-transformed function boundaries - it provides a user with the freedom to declare subfields of a class as "static" (meaning, the value of the field cannot be a JAX traced value, it must be a Python literal, or a constant array - and the value is embedded in the `PyTreeDef` of any instance) or "static" (meaning, the value may be a JAX traced value).
"""


import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import rich.tree as rich_tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    List,
    Tuple,
    static_check_is_array,
    static_check_is_concrete,
    static_check_supports_grad,
)


class Pytree(eqx.Module):
    """`Pytree` is an abstract base class which registers a class with JAX's `Pytree`
    system.
    """

    @classmethod
    def static(cls, **kwargs):
        return eqx.field(**kwargs, static=True)

    @classmethod
    def field(cls, **kwargs):
        return eqx.field(**kwargs)

    def tree_flatten(self):
        return jtu.tree_flatten(self)

    def tree_leaves(self):
        return jtu.tree_leaves(self)

    # This exposes slicing into the struct-of-array representation,
    # taking leaves and indexing into them on the provided index,
    # returning a value with the same `Pytree` structure.
    def slice(self, index_or_index_array: ArrayLike) -> "Pytree":
        """Utility available to any class which mixes `Pytree` base. This
        method supports indexing/slicing on indices when leaves are arrays.

        `obj.slice(index)` will take an instance whose class extends `Pytree`, and return an instance of the same class type, but with leaves indexed into at `index`.

        Arguments:
            index_or_index_array: An `Int` index or an array of indices which will be used to index into the leaf arrays of the `Pytree` instance.

        Returns:
            new_instance: A `Pytree` instance of the same type, whose leaf values are the results of indexing into the leaf arrays with `index_or_index_array`.
        """
        return jtu.tree_map(lambda v: v[index_or_index_array], self)

    ###################
    # Pretty printing #
    ###################

    # Can be customized by Pytree mixers.
    def __rich_tree__(self):
        return gpp.tree_pformat(self)

    # Defines default pretty printing.
    def __rich_console__(self, console, options):
        yield self.__rich_tree__()

    ##############################
    # Utility class constructors #
    ##############################

    @classmethod
    def const(cls, v):
        # The value must be concrete!
        # It cannot be a JAX traced value.
        assert static_check_is_concrete(v)
        if isinstance(v, PytreeConst):
            return v
        else:
            return PytreeConst(v)

    # Safe: will not wrap a PytreeConst in another PytreeConst, and will not
    # wrap dynamic values.
    @classmethod
    def tree_const(cls, v):
        def _inner(v):
            if isinstance(v, PytreeConst):
                return v
            elif static_check_is_concrete(v):
                return PytreeConst(v)
            else:
                return v

        return jtu.tree_map(
            _inner,
            v,
            is_leaf=lambda v: isinstance(v, PytreeConst),
        )

    @classmethod
    def tree_unwrap_const(cls, v):
        def _inner(v):
            if isinstance(v, PytreeConst):
                return v.const
            else:
                return v

        return jtu.tree_map(
            _inner,
            v,
            is_leaf=lambda v: isinstance(v, PytreeConst),
        )

    @classmethod
    def dynamic_closure(cls, fn, *args):
        return DynamicClosure(args, fn)

    #################
    # Static checks #
    #################

    @classmethod
    def static_check_tree_structure_equivalence(cls, trees: List):
        if not trees:
            return True
        else:
            fst, *rest = trees
            treedef = jtu.tree_structure(fst)
            check = all(map(lambda v: treedef == jtu.tree_structure(v), rest))
            return check

    @classmethod
    def static_check_tree_leaves_have_matching_leading_dim(cls, tree):
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

    #############
    # Utilities #
    #############

    @classmethod
    def tree_stack(cls, trees):
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
        result_leaves = [
            jnp.squeeze(jnp.stack(leaf, axis=-1)) for leaf in grouped_leaves
        ]
        return treedef_list[0].unflatten(result_leaves)

    @classmethod
    def tree_unstack(cls, tree):
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

    @classmethod
    def tree_grad_split(cls, tree):
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

    @classmethod
    def tree_grad_zip(cls, grad, nograd):
        def _zipper(*args):
            for arg in args:
                if arg is not None:
                    return arg
            return None

        def _is_none(x):
            return x is None

        return jtu.tree_map(_zipper, grad, nograd, is_leaf=_is_none)


##############################
# Associated utility classes #
##############################


# Wrapper for static values.
class PytreeConst(Pytree):
    const: Any = Pytree.static()

    def __rich_tree__(self):
        return rich_tree.Tree(f"[bold](PytreeConst) {self.const}")


# Construct for a type of closure which closes over dynamic values.
# NOTE: experimental.
class DynamicClosure(Pytree):
    dyn_args: Tuple
    fn: Callable = Pytree.static()

    def __call__(self, *args):
        return self.fn(*self.dyn_args, *args)
