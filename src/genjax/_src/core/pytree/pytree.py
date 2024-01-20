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


import equinox as eqx
import jax.tree_util as jtu

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.pytree.utilities import tree_stack, tree_unstack
from genjax._src.core.typing import ArrayLike


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

    def flatten(self):
        return jtu.tree_leaves(self)

    # This exposes slicing the struct-of-array representation,
    # taking leaves and indexing/randing into them on the first index,
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
