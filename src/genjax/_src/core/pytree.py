# Copyright 2024 MIT Probabilistic Computing Project
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
"""This module contains an abstract data class (called `Pytree`) for implementing JAX's [`Pytree` interface](https://jax.readthedocs.io/en/latest/pytrees.html) on derived classes.

The Pytree interface determines how data classes behave across JAX-transformed function boundaries - it provides a user with the freedom to declare subfields of a class as "static" (meaning, the value of the field cannot be a JAX traced value, it must be a Python literal, or a constant array - and the value is embedded in the `PyTreeDef` of any instance) or "dynamic" (meaning, the value may be a JAX traced value).
"""

import inspect
from dataclasses import field

import jax.numpy as jnp
import jax.tree_util as jtu
from penzai import pz
from penzai.treescope import default_renderer
from penzai.treescope.foldable_representation import (
    basic_parts,
    common_structures,
    common_styles,
    foldable_impl,
)
from penzai.treescope.handlers.penzai import struct_handler
from typing_extensions import dataclass_transform

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


class Pytree(pz.Struct):
    """`Pytree` is an abstract base class which registers a class with JAX's `Pytree`
    system."""

    @dataclass_transform(
        frozen_default=True,
    )
    @classmethod
    def dataclass(
        cls,
        incoming: type[Any] | None = None,
        /,
        **kwargs,
    ) -> type[Any] | Callable[[type[Any]], type[Any]]:
        return pz.pytree_dataclass(
            incoming,
            **kwargs,
        )

    @staticmethod
    def static(**kwargs):
        return field(metadata={"pytree_node": False}, **kwargs)

    @staticmethod
    def field(**kwargs):
        return field(**kwargs)

    # This exposes slicing into the struct-of-array representation,
    # taking leaves and indexing into them on the provided index,
    # returning a value with the same `Pytree` structure.
    def slice(self, index_or_index_array: ArrayLike) -> "Pytree":
        """Utility available to any class which mixes `Pytree` base. This method
        supports indexing/slicing on indices when leaves are arrays.

        `obj.slice(index)` will take an instance whose class extends `Pytree`, and return an instance of the same class type, but with leaves indexed into at `index`.

        Arguments:
            index_or_index_array: An `Int` index or an array of indices which will be used to index into the leaf arrays of the `Pytree` instance.

        Returns:
            new_instance: A `Pytree` instance of the same type, whose leaf values are the results of indexing into the leaf arrays with `index_or_index_array`.
        """
        return jtu.tree_map(lambda v: v[index_or_index_array], self)

    ##############################
    # Utility class constructors #
    ##############################

    @staticmethod
    def const(v):
        # The value must be concrete!
        # It cannot be a JAX traced value.
        assert static_check_is_concrete(v)
        if isinstance(v, Const):
            return v
        else:
            return Const(v)

    # Safe: will not wrap a Const in another Const, and will not
    # wrap dynamic values.
    @staticmethod
    def tree_const(v):
        def _inner(v):
            if isinstance(v, Const):
                return v
            elif static_check_is_concrete(v):
                return Const(v)
            else:
                return v

        return jtu.tree_map(
            _inner,
            v,
            is_leaf=lambda v: isinstance(v, Const),
        )

    @staticmethod
    def tree_unwrap_const(v):
        def _inner(v):
            if isinstance(v, Const):
                return v.const
            else:
                return v

        return jtu.tree_map(
            _inner,
            v,
            is_leaf=lambda v: isinstance(v, Const),
        )

    @staticmethod
    def partial(*args):
        return lambda fn: Closure(args, fn)

    #################
    # Static checks #
    #################

    @staticmethod
    def static_check_tree_structure_equivalence(trees: List):
        if not trees:
            return True
        else:
            fst, *rest = trees
            treedef = jtu.tree_structure(fst)
            check = all(map(lambda v: treedef == jtu.tree_structure(v), rest))
            return check

    @staticmethod
    def static_check_none(v):
        return v == Const(None)

    @staticmethod
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

    #############
    # Utilities #
    #############

    @staticmethod
    def tree_stack(trees):
        """Takes a list of trees and stacks every corresponding leaf.

        For example, given two trees ((a, b), c) and ((a', b'), c'), returns ((stack(a,
        a'), stack(b, b')), stack(c, c')).

        Useful for turning a list of objects into something you can feed to a vmapped
        function.
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def tree_grad_zip(grad, nograd):
        def _zipper(*args):
            for arg in args:
                if arg is not None:
                    return arg
            return None

        def _is_none(x):
            return x is None

        return jtu.tree_map(_zipper, grad, nograd, is_leaf=_is_none)

    def pprint(self):
        with pz.ts.active_autovisualizer.set_scoped(pz.ts.ArrayAutovisualizer()):
            pz.ts.display(self)

    def render_html(self):
        def custom_handler(node, path, subtree_renderer):
            if inspect.isfunction(node):
                mod_path = node.__module__
                return basic_parts.siblings_with_annotations(
                    common_structures.build_one_line_tree_node(
                        line=common_styles.AbbreviationColor(
                            basic_parts.Text(f"<fn {node.__name__}>")
                        ),
                        path=None,
                    ),
                    foldable_impl.StringCopyButton(mod_path),
                )
            if isinstance(node, Pytree):
                mod_path = node.__module__
                inner = struct_handler.handle_structs(node, None, subtree_renderer)
                return basic_parts.siblings_with_annotations(
                    inner,
                    foldable_impl.StringCopyButton(
                        annotation="Module path: ", copy_string=mod_path
                    ),
                )
            return NotImplemented

        default_renderer.active_renderer.get().handlers.insert(0, custom_handler)
        return pz.ts.render_to_html(
            self,
            roundtrip_mode=False,
        )


##############################
# Associated utility classes #
##############################


# Wrapper for static values (can include callables).
@Pytree.dataclass
class Const(Pytree):
    const: Any = Pytree.static()

    def __call__(self, *args):
        return self.const(*args)


# Construct for a type of closure which closes over dynamic values.
# NOTE: experimental.
@Pytree.dataclass
class Closure(Pytree):
    dyn_args: Tuple
    fn: Callable = Pytree.static()

    def __call__(self, *args):
        return self.fn(*self.dyn_args, *args)
