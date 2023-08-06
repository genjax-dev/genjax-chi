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
"""This module contains a utility class for defining new `jax.Pytree`
implementors.

In addition to this functionality, there's a "sum type" `Pytree`
implementation which allows effective decomposition of multiple
potential `Pytree` value inhabitants into a common tree shape.

This allows, among other things, an efficient implementation of
`SwitchCombinator`.
"""

import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import api_util

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.hashable_dict import HashableDict
from genjax._src.core.datatypes.hashable_dict import hashable_dict
from genjax._src.core.interpreters.staging import is_concrete
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Sequence
from genjax._src.core.typing import Tuple
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

    This function respects concrete vs. traced values. It will leave
    concrete leaves unchanged (it will not lift them to
    `jax.core.Tracer`).
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


#####
# Pytree abstract base
#####


class Pytree(metaclass=abc.ABCMeta):
    """> Abstract base class which registers a class with JAX's `Pytree`
    system.

    Users who mixin this ABC for class definitions are required to
    implement `flatten` below. In turn, instances of the class gain
    access to a large set of utility functions for working with `Pytree`
    data, as well as the ability to use `jax.tree_util` Pytree
    functionality.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_node(
            cls,
            cls.flatten,
            cls.unflatten,
        )

    @abc.abstractmethod
    def flatten(self) -> Tuple[Tuple, Tuple]:
        """`flatten` must be implemented when a user mixes `Pytree` into the
        declaration of a new class or dataclass.

        The implementation of `flatten` assumes the following contract:

        * must return a 2-tuple of tuples.
        * the first tuple is "dynamic" data - things that JAX tracers are allowed to population.
        * the second tuple is "static" data - things which are known at JAX tracing time. Static data is also used by JAX for `Pytree` equality comparison.

        For more information, consider [JAX's documentation on Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html).

        Returns:

            dynamic: Dynamic data which supports JAX tracer values.
            static: Static data which is JAX trace time constant.

        Examples:

            Let's assume that you are implementing a new dataclass. Here's how you would define the dataclass using the `Pytree` mixin.

            ```python
            @dataclass
            class MyFoo(Pytree):
                static_field: Any
                dynamic_field: Any

                # Implementing `flatten`
                def flatten(self):
                    return (self.dynamic_field, ), (self.static_field, )
            ```

            !!! info "Ordering fields in `Pytree` declarations"

                Note that the ordering in the dataclass declaration **does matter** - you should put static fields first. The automatically defined `unflatten` method (c.f. below) assumes this ordering.

            Now, given the declaration, you can use `jax.tree_util` flattening/unflatten functionality.

            ```python exec="yes" source="tabbed-left"
            import genjax
            import jax.tree_util as jtu
            from genjax.core import Pytree
            from dataclasses import dataclass
            console = genjax.pretty()

            @dataclass
            class MyFoo(Pytree):
                static_field: Any
                dynamic_field: Any

                # Implementing `flatten`
                def flatten(self):
                    return (self.dynamic_field, ), (self.static_field, )

            f = MyFoo(0, 1.0)
            leaves, form = jtu.tree_flatten(f)

            print(console.render(leaves))
            new = jtu.tree_unflatten(form, leaves)
            print(console.render(new))
            ```
        """

    @classmethod
    def unflatten(cls, data, xs):
        """Given an implementation of `flatten` (c.f. above), `unflatten` is
        automatically defined and registered with JAX's `Pytree` system.

        `unflatten` allows usage of `jtu.tree_unflatten` to create instances of a declared class that mixes `Pytree` from a `PyTreeDef` for that class and leaf data.

        Examples:

            Our example from `flatten` above also applies here - where we use `jtu.tree_unflatten` to create a new instance of `MyFoo` from a `PyTreeDef` and leaf data.

            ```python exec="yes" source="tabbed-left"
            import genjax
            import jax.tree_util as jtu
            from genjax.core import Pytree
            from dataclasses import dataclass
            console = genjax.pretty()

            @dataclass
            class MyFoo(Pytree):
                static_field: Any
                dynamic_field: Any

                # Implementing `flatten`
                def flatten(self):
                    return (self.dynamic_field, ), (self.static_field, )

            f = MyFoo(0, 1.0)
            leaves, form = jtu.tree_flatten(f)

            new = jtu.tree_unflatten(form, leaves)
            print(console.render(new))
            ```
        """
        return cls(*data, *xs)

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    # This exposes slicing the struct-of-array representation,
    # taking leaves and indexing/randing into them on the first index,
    # returning a value with the same `Pytree` structure.
    def slice(self, index_or_index_array):
        """> Utility available to any class which mixes `Pytree` base. This
        method supports indexing/slicing on indices when leaves are arrays.

        `obj.slice(index)` will take an instance whose class extends `Pytree`, and return an instance of the same class type, but with leaves indexed into at `index`.

        Arguments:

            index_or_index_array: An `Int` index or an array of indices which will be used to index into the leaf arrays of the `Pytree` instance.

        Returns:

            new_instance: A `Pytree` instance of the same type, whose leaf values are the results of indexing into the leaf arrays with `index_or_index_array`.
        """
        return jtu.tree_map(lambda v: v[index_or_index_array], self)

    def stack(self, *trees):
        return tree_stack([self, *trees])

    def unstack(self):
        return tree_unstack(self)

    # Lift multiple trees into a sum type.
    def sum(self, *trees):
        return Sumtree.new(self, trees)

    # Can be customized by Pytree mixers.
    def __rich_tree__(self, tree):
        sub_tree = gpp.tree_pformat(self)
        tree.add(sub_tree)
        return tree

    # Defines default pretty printing.
    def __rich_console__(self, console, options):
        tree = gpp.tree_pformat(self)
        yield tree

    def __rich_repr__(self):
        yield self


#####
# Dynamic closure
#####


@dataclass
class DynamicClosure(Pytree):
    fn: Callable
    dyn_args: Tuple

    def flatten(self):
        return (self.dyn_args,), (self.fn,)

    @classmethod
    def new(cls, callable, *dyn_args):
        if isinstance(callable, DynamicClosure):
            return DynamicClosure(callable.fn, (*callable.dyn_args, *dyn_args))
        else:
            return DynamicClosure(callable, dyn_args)

    def __call__(self, *args):
        return self.fn(*self.dyn_args, *args)

    def __hash__(self):
        avals = tuple(api_util.shaped_abstractify(i) for i in self.dyn_args)
        return hash((self.fn, *avals))


def dynamic_closure(*args):
    return lambda fn: DynamicClosure.new(fn, *args)


#####
# Pytree sum type
#####

# If you have multiple Pytrees, you might want
# to generate a "sum" Pytree with leaves that minimally cover
# the entire set of dtypes and shapes.
#
# The code below is intended to provide this functionality.


def get_call_fallback(d, k, fn, fallback):
    if k in d:
        d[k] = fn(d[k])
    else:
        d[k] = fallback


def minimum_covering_leaves(pytrees: Sequence):
    leaf_schema = hashable_dict()
    for tree in pytrees:
        local = hashable_dict()
        jtu.tree_map(
            lambda v: get_call_fallback(local, v, lambda v: v + 1, 1),
            tree,
        )
        for (k, v) in local.items():
            get_call_fallback(leaf_schema, k, lambda u: v if v > u else u, v)

    return leaf_schema


def shape_dtype_struct(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def set_payload(leaf_schema, pytree):
    leaves = jtu.tree_leaves(pytree)
    payload = hashable_dict()
    for k in leaves:
        aval = shape_dtype_struct(jax.core.get_aval(k))
        if aval in payload:
            shared = payload[aval]
        else:
            shared = []
            payload[aval] = shared
        shared.append(k)

    for (k, limit) in leaf_schema.items():
        dtype = k.dtype
        shape = k.shape
        if k in payload:
            v = payload[k]
            cur_len = len(v)
            v.extend([jnp.zeros(shape, dtype) for _ in range(0, limit - cur_len)])
        else:
            payload[k] = [jnp.zeros(shape, dtype) for _ in range(0, limit)]
    return payload


def get_visitation(pytree):
    return jtu.tree_flatten(pytree)


def build_from_payload(visitation, form, payload):
    counter = hashable_dict()

    def _check_counter_get(k):
        index = counter.get(k, 0)
        counter[k] = index + 1
        return payload[k][index]

    payload_copy = [_check_counter_get(k) for k in visitation]
    return jtu.tree_unflatten(form, payload_copy)


@dataclass
class StaticCollection(Pytree):
    seq: Sequence

    def flatten(self):
        return (), (self.seq,)


@dataclass
class Sumtree(Pytree):
    visitations: StaticCollection
    forms: StaticCollection
    payload: HashableDict

    def flatten(self):
        return (self.payload,), (self.visitations, self.forms)

    @classmethod
    def new(cls, source: Pytree, covers: Sequence[Pytree]):
        leaf_schema = minimum_covering_leaves(covers)
        visitations = []
        forms = []
        for cover in covers:
            visitation, form = get_visitation(cover)
            visitations.append(visitation)
            forms.append(form)
        visitations = StaticCollection(visitations)
        forms = StaticCollection(forms)
        payload = set_payload(leaf_schema, source)
        return Sumtree(visitations, forms, payload)

    def materialize_iterator(self):
        static_visitations = self.visitations.seq
        static_forms = self.forms.seq
        return map(
            lambda args: build_from_payload(args[0], args[1], self.payload),
            zip(static_visitations, static_forms),
        )

    # Collapse the sum type.
    def project(self):
        static_visitations = self.visitations.seq
        static_forms = self.forms.seq
        return build_from_payload(
            static_visitations[0],
            static_forms[0],
            self.payload,
        )
