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
from genjax.builtin.datatypes import JAXChoiceMap
from dataclasses import dataclass

#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    subtrace: Trace
    length: int

    def __init__(self, subtrace, length):
        if isinstance(subtrace, dict):
            self.subtrace = JAXChoiceMap(subtrace)
        else:
            self.subtrace = subtrace
        self.length = length

    def flatten(self):
        return (self.subtrace,), (self.length,)

    @classmethod
    def unflatten(cls, data, xs):
        return VectorChoiceMap(*xs, *data)

    def get_choice(self, addr):
        return self.subtrace.get_choice(addr)

    def has_choice(self, addr):
        return self.subtrace.has_choice(addr)

    def has_value(self):
        return False

    def get_value(self):
        raise Exception("VectorChoiceMap is not a value choice map.")

    # TODO.
    def get_choices_shallow(self):
        return ()


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
        chm_vectored.append(emptied)
        mask_vectored.append(mask)
    mask_vectored = tree_stack(mask_vectored)
    chm_vectored = tree_stack(chm_vectored)
    return chm_vectored, mask_vectored
