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
"""This module contains a utility class for defining new `jax.Pytree`
implementors."""

import abc

import jax.tree_util as jtu

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.pytree.utilities import tree_stack, tree_unstack
from genjax._src.core.typing import Tuple


class Pytree:
    """`Pytree` is an abstract base class which registers a class with JAX's `Pytree`
    system.

    Users who mixin this ABC for class definitions are required to
    implement `flatten` below. In turn, instances of the class gain
    access to a large set of utility functions for working with `Pytree`
    data, as well as the ability to use `jax.tree_util` Pytree
    functionality.
    """

    def __init_subclass__(cls, **kwargs):
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
            console = genjax.console()

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
            console = genjax.console()

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

    # This exposes slicing the struct-of-array representation,
    # taking leaves and indexing/randing into them on the first index,
    # returning a value with the same `Pytree` structure.
    def slice(self, index_or_index_array):
        """Utility available to any class which mixes `Pytree` base. This
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

    ###################
    # Pretty printing #
    ###################

    # Can be customized by Pytree mixers.
    def __rich_tree__(self):
        return gpp.tree_pformat(self)

    # Defines default pretty printing.
    def __rich_console__(self, console, options):
        yield self.__rich_tree__()
