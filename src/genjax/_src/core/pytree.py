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

from dataclasses import field

import jax.numpy as jnp
import jax.tree_util as jtu
from penzai import pz
from typing_extensions import dataclass_transform

from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    List,
    Tuple,
    static_check_is_array,
    static_check_is_concrete,
    static_check_supports_grad,
)

register_exclusion(__file__)


class Pytree(pz.Struct):
    """`Pytree` is an abstract base class which registers a class with JAX's `Pytree`
    system. JAX's `Pytree` system tracks how data classes should behave across JAX-transformed function boundaries, like `jax.jit` or `jax.vmap`.

    Inheriting this class provides the implementor with the freedom to declare how the subfields of a class should behave:

    * `Pytree.static(...)`: the value of the field cannot be a JAX traced value, it must be a Python literal, or a constant). The values of static fields are embedded in the `PyTreeDef` of any instance of the class.
    * `Pytree.field(...)` or no annotation: the value may be a JAX traced value, and JAX will attempt to convert it to tracer values inside of its transformations.

    If a field _points to another `Pytree`_, it should not be declared as `Pytree.static()`, as the `Pytree` interface will automatically handle the `Pytree` fields as dynamic fields.

    """

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
        """
        Denote that a class (which is inheriting `Pytree`) should be treated as a dataclass, meaning it can hold data in fields which are declared as part of the class.

        A dataclass is to be distinguished from a "methods only" `Pytree` class, which does not have fields, but may define methods.
        The latter cannot be instantiated, but can be inherited from, while the former can be instantiated:
        the `Pytree.dataclass` declaration informs the system _how to instantiate_ the class as a dataclass,
        and how to automatically define JAX's `Pytree` interfaces (`tree_flatten`, `tree_unflatten`, etc.) for the dataclass, based on the fields declared in the class, and possibly `Pytree.static(...)` or `Pytree.field(...)` annotations (or lack thereof, the default is that all fields are `Pytree.field(...)`).

        All `Pytree` dataclasses support pretty printing, as well as rendering to HTML.

        Example:
            ```python exec="yes" html="true" source="material-block" session="core"
            from genjax import Pytree
            from genjax.typing import FloatArray, typecheck
            import jax.numpy as jnp

            @Pytree.dataclass
            @typecheck # Enforces type annotations on instantiation.
            class MyClass(Pytree):
                my_static_field: int = Pytree.static()
                my_dynamic_field: FloatArray

            print(MyClass(10, jnp.array(5.0)).render_html())
            ```
        """

        return pz.pytree_dataclass(
            incoming,
            **kwargs,
        )

    @staticmethod
    def static(**kwargs):
        """Declare a field of a `Pytree` dataclass to be static. Users can provide additional keyword argument options,
        like `default` or `default_factory`, to customize how the field is instantiated when an instance of
        the dataclass is instantiated.` Fields which are provided with default values must come after required fields in the dataclass declaration.

        Example:
            ```python exec="yes" html="true" source="material-block" session="core"
            @Pytree.dataclass
            @typecheck # Enforces type annotations on instantiation.
            class MyClass(Pytree):
                my_dynamic_field: FloatArray
                my_static_field: int = Pytree.static(default=0)

            print(MyClass(jnp.array(5.0)).render_html())
            ```

        """
        return field(metadata={"pytree_node": False}, **kwargs)

    @staticmethod
    def field(**kwargs):
        "Declare a field of a `Pytree` dataclass to be dynamic. Alternatively, one can leave the annotation off in the declaration."
        return field(**kwargs)

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

    def treedef(self):
        return jtu.tree_structure(self)

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

    def __call__(self, *args, **kwargs):
        return self.fn(*self.dyn_args, *args, **kwargs)
