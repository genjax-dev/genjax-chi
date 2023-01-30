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

from dataclasses import dataclass
from typing import Union

import jax.tree_util as jtu
import numpy as np
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import EmptyChoiceMap
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import Trace
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import typecheck


######################################
# Vector-shaped combinator datatypes #
######################################

# This section applies to `Map` and `Unfold`, currently.

#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    indices: IntArray
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.indices, self.inner), ()

    @typecheck
    @classmethod
    def new(cls, inner: ChoiceMap):
        if isinstance(inner, EmptyChoiceMap):
            return inner
        broadcast_dim_tree = jtu.tree_map(lambda v: len(v), inner)
        leaves = jtu.tree_leaves(broadcast_dim_tree)
        leaf_lengths = set(leaves)
        assert len(leaf_lengths) == 1
        max_index = list(leaf_lengths).pop()
        return VectorChoiceMap(np.arange(0, max_index), inner)

    def get_selection(self):
        subselection = self.inner.get_selection()
        return VectorSelection.new(self.indices, subselection)

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return self.inner.get_subtree(addr)

    def get_subtrees_shallow(self):
        return self.inner.get_subtrees_shallow()

    # TODO: This currently provides poor support for merging
    # two vector choices maps with different index arrays.
    def merge(self, other):
        if isinstance(other, VectorChoiceMap):
            return VectorChoiceMap(other.indices, self.inner.merge(other.inner))
        else:
            return VectorChoiceMap(
                self.indices,
                self.inner.merge(other),
            )

    def __hash__(self):
        return hash(self.inner)

    def get_index(self):
        return self.indices

    def _tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        subt = self.inner._build_rich_tree()
        subk = Tree("[blue]indices")
        subk.add(gpp.tree_pformat(self.indices))
        tree.add(subk)
        tree.add(subt)
        return tree


#####
# VectorSelection
#####


@dataclass
class VectorSelection(Selection):
    pass


##############
# Shorthands #
##############

vector_choice_map = VectorChoiceMap.new
