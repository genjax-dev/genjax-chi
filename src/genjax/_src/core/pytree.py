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
from dataclasses import field, fields
from typing import overload

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
from penzai.treescope.handlers import builtin_structure_handler
from penzai.treescope.handlers.penzai import struct_handler
from typing_extensions import dataclass_transform

from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    Int,
    TypeVar,
    static_check_is_array,
    static_check_is_concrete,
    static_check_supports_grad,
)

register_exclusion(__file__)

R = TypeVar("R")
_T = TypeVar("_T")


class Pytree(pz.Struct):
    """`Pytree` is an abstract base class which registers a class with JAX's `Pytree`
    system. JAX's `Pytree` system tracks how data classes should behave across JAX-transformed function boundaries, like `jax.jit` or `jax.vmap`.

    Inheriting this class provides the implementor with the freedom to declare how the subfields of a class should behave:

    * `Pytree.static(...)`: the value of the field cannot be a JAX traced value, it must be a Python literal, or a constant). The values of static fields are embedded in the `PyTreeDef` of any instance of the class.
    * `Pytree.field(...)` or no annotation: the value may be a JAX traced value, and JAX will attempt to convert it to tracer values inside of its transformations.

    If a field _points to another `Pytree`_, it should not be declared as `Pytree.static()`, as the `Pytree` interface will automatically handle the `Pytree` fields as dynamic fields.

    """

    @classmethod
    @overload
    def dataclass(
        cls,
        incoming: None = None,
        /,
        **kwargs,
    ) -> Callable[[type[_T]], type[_T]]: ...

    @classmethod
    @overload
    def dataclass(
        cls,
        incoming: type[_T],
        /,
        **kwargs,
    ) -> type[_T]: ...

    @dataclass_transform(
        frozen_default=True,
    )
    @classmethod
    def dataclass(
        cls,
        incoming: type[_T] | None = None,
        /,
        **kwargs,
    ) -> type[_T] | Callable[[type[_T]], type[_T]]:
        """
        Denote that a class (which is inheriting `Pytree`) should be treated as a dataclass, meaning it can hold data in fields which are declared as part of the class.

        A dataclass is to be distinguished from a "methods only" `Pytree` class, which does not have fields, but may define methods.
        The latter cannot be instantiated, but can be inherited from, while the former can be instantiated:
        the `Pytree.dataclass` declaration informs the system _how to instantiate_ the class as a dataclass,
        and how to automatically define JAX's `Pytree` interfaces (`tree_flatten`, `tree_unflatten`, etc.) for the dataclass, based on the fields declared in the class, and possibly `Pytree.static(...)` or `Pytree.field(...)` annotations (or lack thereof, the default is that all fields are `Pytree.field(...)`).

        All `Pytree` dataclasses support pretty printing, as well as rendering to HTML.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="core"
            from genjax import Pytree
            from genjax.typing import FloatArray, typecheck
            import jax.numpy as jnp


            @Pytree.dataclass
            @typecheck  # Enforces type annotations on instantiation.
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

        Examples:
            ```python exec="yes" html="true" source="material-block" session="core"
            @Pytree.dataclass
            @typecheck  # Enforces type annotations on instantiation.
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
    def partial(*args) -> Callable[[Callable[..., R]], "Closure[R]"]:
        return lambda fn: Closure[R](args, fn)

    def treedef(self):
        return jtu.tree_structure(self)

    #################
    # Static checks #
    #################

    @staticmethod
    def static_check_tree_structure_equivalence(trees: list):
        if not trees:
            return True
        else:
            fst, *rest = trees
            treedef = jtu.tree_structure(fst)
            check = all(map(lambda v: treedef == jtu.tree_structure(v), rest))
            return check

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

    def render_html(self):
        def _pytree_handler(node, subtree_renderer):
            constructor_open = struct_handler.render_struct_constructor(node)
            fs = fields(node)

            (
                background_color,
                background_pattern,
            ) = builtin_structure_handler.parse_color_and_pattern(
                node.treescope_color(), type(node).__name__
            )

            if background_pattern is not None:
                if background_color is None:
                    raise ValueError(
                        "background_color must be provided if background_pattern is"
                    )

                def wrapper1(block):
                    return common_styles.WithBlockPattern(
                        block, color=background_color, pattern=background_pattern
                    )

                wrap_block = wrapper1
                wrap_topline = common_styles.PatternedTopLineSpanGroup
                wrap_bottomline = common_styles.PatternedBottomLineSpanGroup

            elif background_color is not None and background_color != "transparent":

                def wrapper2(block):
                    return common_styles.WithBlockColor(block, color=background_color)

                wrap_block = wrapper2
                wrap_topline = common_styles.ColoredTopLineSpanGroup
                wrap_bottomline = common_styles.ColoredBottomLineSpanGroup

            else:

                def id(rendering):
                    return rendering

                wrap_block = id
                wrap_topline = id
                wrap_bottomline = id

            children = builtin_structure_handler.build_field_children(
                node,
                None,
                subtree_renderer,
                fields_or_attribute_names=fs,
                key_path_fn=node.key_for_field,
                attr_style_fn=struct_handler.struct_attr_style_fn_for_fields(fs),
            )
            children = basic_parts.IndentedChildren(children)

            suffix = ")"

            return wrap_block(
                basic_parts.Siblings(
                    children=[
                        wrap_topline(constructor_open),
                        basic_parts.Siblings.build(
                            foldable_impl.HyperlinkTarget(
                                foldable_impl.FoldableTreeNodeImpl(
                                    basic_parts.FoldCondition(
                                        collapsed=basic_parts.Text("..."),
                                        expanded=children,
                                    )
                                ),
                                keypath=None,
                            ),
                            wrap_bottomline(basic_parts.Text(suffix)),
                        ),
                    ],
                )
            )

        def custom_handler(node, path, subtree_renderer):
            if inspect.isfunction(node):
                return common_structures.build_one_line_tree_node(
                    line=common_styles.CustomTextColor(
                        basic_parts.Text(f"<fn {node.__name__}>"),
                        color="blue",
                    ),
                    path=None,
                )
            if isinstance(node, Pytree):
                return _pytree_handler(node, subtree_renderer)
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
    """
    JAX-compatible way to tag a value as a constant. Valid constants include Python literals, strings, essentially anything **that won't hold JAX arrays** inside of a computation.

    Examples:
        Instances of `Const` can be created using a `Pytree` classmethod:
        ```python exec="yes" html="true" source="material-block" session="core"
        from genjax import Pytree

        c = Pytree.const(5)
        print(c.render_html())
        ```

        Constants can be freely used across [`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) boundaries:
        ```python exec="yes" html="true" source="material-block" session="core"
        from genjax import Pytree


        def f(c):
            if c.const == 5:
                return 10.0
            else:
                return 5.0


        c = Pytree.const(5)
        r = jax.jit(f)(c)
        print(r)
        ```
    """

    const: Any = Pytree.static()

    def __call__(self, *args):
        return self.const(*args)


# Construct for a type of closure which closes over dynamic values.
@Pytree.dataclass
class Closure(Generic[R], Pytree):
    """
    JAX-compatible closure type. It's a closure _as a [`Pytree`][genjax.core.Pytree]_ - meaning the static _source code_ / _callable_ is separated from dynamic data (which must be tracked by JAX).

    Examples:
        Instances of `Closure` can be created using `Pytree.partial` -- note the order of the "closed over" arguments:
        ```python exec="yes" html="true" source="material-block" session="core"
        from genjax import Pytree


        def g(y):
            @Pytree.partial(y)  # dynamic values come first
            def f(v, x):
                # v will be bound to the value of y
                return x * (v * 5.0)

            return f


        clos = jax.jit(g)(5.0)
        print(clos.render_html())
        ```

        Closures can be invoked / JIT compiled in other code:
        ```python exec="yes" html="true" source="material-block" session="core"
        r = jax.jit(lambda x: clos(x))(3.0)
        print(r)
        ```
    """

    dyn_args: tuple
    fn: Callable[..., R] = Pytree.static()

    def __call__(self, *args, **kwargs) -> R:
        return self.fn(*self.dyn_args, *args, **kwargs)


def nth(x: Pytree, idx: Int):
    """Returns a Pytree in which `[idx]` has been applied to every leaf."""
    return jtu.tree_map(lambda v: v[idx], x)


class PythonicPytree(Pytree):
    """
    A class that adds support for bracket indexing/slicing, sequence-like operations,
    and concatenation to make working with Pytrees more Pythonic. The base class is
    appropriate for Pytrees which have a unform shape over leaves (or at least each
    leaf's initial axis should have the same length).
    """

    def __getitem__(self, idx):
        """Return a pytree in which each leaf has been sliced by `idx`."""
        return nth(self, idx)

    def __len__(self):
        """Return the "length" of the Pytree. This should only be used on
        Pytrees which have a uniform shape over leaves; it operates by
        returning the length of the "first" leaf."""
        return len(jtu.tree_leaves(self)[0])

    def __iter__(self):
        """Returs an iterator which generates each self[i] in order"""
        return (self[i] for i in range(len(self)))

    def __add__(self, other):
        """Concatenates two pytrees, leaf-wise."""
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")

        def concat_leaves(x, y):
            return jnp.concatenate([x, y])

        return jtu.tree_map(concat_leaves, self, other)

    def prepend(self, child):
        """Prepends a scalar element to the front of each leaf in a Pytree."""
        return jtu.tree_map(lambda x: x[jnp.newaxis], child) + self
