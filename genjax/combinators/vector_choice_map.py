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

"""
This module holds a common :code:`ChoiceMap` implementation for
:code:`genjax.MapCombinator` and :code:`genjax.UnfoldCombinator`.

This module also holds a static utility function, which is used in the
generative function implementations for the above combinators to
coerce hierarchical choice maps into forms which can be :code:`scan`d
and :code:`vmap`d.
"""

import jax.tree_util as jtu
import numpy as np
from genjax.core.datatypes import ChoiceMap, Trace
from genjax.core.pytree import tree_stack
from genjax.distributions.distribution import ValueChoiceMap
from dataclasses import dataclass
from typing import Any

#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    subtrace: Trace
    length: int
    treedef: Any

    def flatten(self):
        return (self.subtrace, self.length), (self.treedef,)

    @classmethod
    def unflatten(cls, data, xs):
        return VectorChoiceMap(*data, *xs)

    def get_choice(self, addr):
        if isinstance(addr, int):
            slice = jtu.tree_map(lambda v: v[addr], self.subtrace)
            return slice
        elif isinstance(addr, tuple):
            index = addr[0]
            assert isinstance(index, int)
            slice = jtu.tree_map(lambda v: v[index], self.subtrace)
            rest = addr[1:]
            return slice.get_choice(rest)

    def has_choice(self, addr):
        if isinstance(addr, int):
            if addr > self.length:
                return False
            else:
                return True

        if not isinstance(addr, tuple) or not isinstance(addr[0], int):
            return False

        index = addr[0]
        if index > self.length:
            return False
        rest = addr[1:]
        return self.subtrace.get_choices().has_choice(rest)

    # TODO.
    def get_choices_shallow(self):
        return ()

    def map(self, fn):
        chm = self.subtrace.get_choices()
        return chm.map(fn)


# This doesn't return an actual `VectorChoiceMap`, but is a utility
# used by both `Unfold` and `Map` to create a broadcasted chm
# and a mask, for use with `DynamicJAXGenerativeFunction`.
def prepare_vectorized_choice_map(shape, treedef, length, chm):
    chm_vectored = []
    mask_vectored = []
    for k in range(0, length):
        emptied = jtu.tree_map(lambda v: np.zeros((), v.dtype), shape)
        if chm.has_choice(k):
            submap = chm.get_choice(k)
            emptied = emptied.merge(submap)
            mask = [
                True if submap.has_choice(k) else False
                for (k, _) in shape.get_choices_shallow()
            ]
            mask = jtu.tree_unflatten(treedef, mask)
        else:
            mask = [False for (k, _) in shape.get_choices_shallow()]
            mask = jtu.tree_unflatten(treedef, mask)
        mask = jtu.tree_map(
            lambda v: v.value,
            mask,
            is_leaf=lambda v: isinstance(v, ValueChoiceMap),
        )
        chm_vectored.append(emptied)
        mask_vectored.append(mask)
    mask_vectored = tree_stack(mask_vectored)
    chm_vectored = tree_stack(chm_vectored)
    return chm_vectored, mask_vectored
