# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the `Mask` datatype backing GenJAX's masking system.

Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as constraints in choice maps, and participate in inference computations (like scores, and importance weights or density ratios).

Masks are also used internally by generative function combinators which include uncertainty over structure.

Users are expected to interact with `Mask` instances by unmasking them using the `Mask.unmask` interface. This interface uses JAX's `checkify` transformation to ensure that masked data exposed to a user is used only when valid. If a user chooses to `Mask.unmask` a `Mask` instance, they are also expected to use `jax.experimental.checkify.checkify` to transform their function to one which could return an error.

Read more: [jax.experimental.checkify.checkify](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html)
"""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import checkify
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.address_tree import AddressLeaf
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import dispatch
from genjax._src.global_options import global_options


@dataclass
class Mask(Pytree):
    mask: BoolArray
    value: Any

    def flatten(self):
        return (self.mask, self.value), ()

    @classmethod
    def new(cls, mask: BoolArray, inner):
        if isinstance(inner, cls):
            return Mask(
                jnp.logical_and(mask, inner.mask),
                inner.value(),
            )
        else:
            return cls(mask, inner).leaf_push()

    def unmask(self):
        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.mask)
            checkify.check(check_flag, "Mask is False, the masked value is invalid.\n")

        global_options.optional_check(_check)
        return self.value

    def unsafe_unmask(self):
        # Unsafe version of unmask -- should only be used internally.
        return self.value

    def _set_leaf(self, v: AddressLeaf):
        leaf_value = v.get_leaf_value()
        if isinstance(leaf_value, Mask):
            leaf_mask = leaf_value.mask
            return v.set_leaf_value(
                Mask(
                    jnp.logical_and(self.mask, leaf_mask),
                    leaf_value.unmask(),
                )
            )
        else:
            return v.set_leaf_value(Mask(self.mask, leaf_value))

    def leaf_push(self):
        def _inner(v):
            if isinstance(v, Mask):
                return Mask.new(self.mask, v.value())

            # `AddressLeaf` inheritors have a method `set_leaf_value`
            # to participate in masking.
            # They can choose how to construct themselves after
            # being provided with a masked value.
            elif isinstance(v, AddressLeaf):
                return self._set_leaf(v)
            else:
                return v

        def _check(v):
            return isinstance(v, Mask) or isinstance(v, AddressLeaf)

        return jtu.tree_map(_inner, self.value, is_leaf=_check)

    ###########
    # Dunders #
    ###########

    @dispatch
    def __eq__(self, other: "Mask"):
        return jnp.logical_and(
            jnp.logical_and(self.mask, other.mask),
            self.value == other.value,
        )

    @dispatch
    def __eq__(self, other: Any):
        return jnp.logical_and(
            self.mask,
            self.value == other,
        )

    def __hash__(self):
        hash1 = hash(self.value)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.mask, short_arrays=True)
        val = gpp._pformat_array(self.value, short_arrays=True)
        sub_tree = Tree(f"[bold](Mask, {doc})")
        sub_tree.add(Tree(f"{val}"))
        tree.add(sub_tree)
        return tree

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


##############
# Shorthands #
##############

mask = Mask.new
