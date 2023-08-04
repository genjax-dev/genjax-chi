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

import jax.numpy as jnp
import jax.tree_util as jtu
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import choice_map
from genjax._src.core.datatypes.generative import select
from genjax._src.core.datatypes.masking import mask
from genjax._src.core.pytree import tree_stack
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import String
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.vector.vector_utilities import (
    static_check_leaf_length,
)


######################################
# Vector-shaped combinator datatypes #
######################################

# The data types in this section are used in `Map` and `Unfold`, currently.

#####
# IndexSelection
#####


@dataclass
class IndexSelection(Selection):
    indices: IntArray
    inner: Selection

    def flatten(self):
        return (
            self.indices,
            self.inner,
        ), ()

    @classmethod
    @dispatch
    def new(cls, idx: Union[Int, IntArray]):
        idxs = jnp.array(idx)
        return IndexSelection(idxs, AllSelection())

    @classmethod
    @dispatch
    def new(cls, idx: Union[Int, List[Int], IntArray], inner: Selection):
        idxs = jnp.array(idx)
        return IndexSelection(idxs, inner)

    @dispatch
    def has_subtree(self, addr: IntArray):
        return jnp.isin(addr, self.indices)

    @dispatch
    def has_subtree(self, addr: Tuple):
        if len(addr) <= 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))

    def get_subtree(self, addr):
        raise NotImplementedError

    def complement(self):
        return ComplementIndexSelection(self)


@dataclass
class ComplementIndexSelection(IndexSelection):
    index_selection: Selection

    def flatten(self):
        return (self.index_selection,), ()

    @dispatch
    def has_subtree(self, addr: IntArray):
        return jnp.logical_not(jnp.isin(addr, self.indices))

    @dispatch
    def has_subtree(self, addr: Tuple):
        if len(addr) <= 1:
            return False
        (idx, addr) = addr
        return jnp.logical_not(
            jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))
        )

    def get_subtree(self, addr):
        raise NotImplementedError

    def complement(self):
        return self.index_selection


#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    @dispatch
    def new(
        cls,
        inner: ChoiceMap,
    ) -> ChoiceMap:
        if isinstance(inner, EmptyChoiceMap):
            return inner
        return VectorChoiceMap(inner)

    @classmethod
    @dispatch
    def new(
        cls,
        inner: Dict,
    ) -> ChoiceMap:
        chm = choice_map(inner)
        return VectorChoiceMap.new(chm)

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        return VectorChoiceMap.new(self.inner.filter(selection))

    @dispatch
    def filter(
        self,
        selection: IndexSelection,
    ) -> ChoiceMap:
        filtered = self.inner.filter(selection.inner)
        flags = jnp.logical_and(
            selection.indices >= 0,
            selection.indices < static_check_leaf_length(self.inner),
        )

        def _take(v):
            return jnp.take(v, selection.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    @dispatch
    def filter(
        self,
        selection: ComplementIndexSelection,
    ) -> ChoiceMap:
        filtered = self.inner.filter(selection.inner.complement())
        flags = jnp.logical_not(
            jnp.logical_and(
                selection.indices >= 0,
                selection.indices < static_check_leaf_length(self.inner),
            )
        )

        def _take(v):
            return jnp.take(v, selection.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    def get_selection(self):
        subselection = self.inner.get_selection()
        return subselection

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return self.inner.get_subtree(addr)

    def get_subtrees_shallow(self):
        return self.inner.get_subtrees_shallow()

    @dispatch
    def merge(self, other: "VectorChoiceMap") -> Tuple[ChoiceMap, ChoiceMap]:
        new, discard = self.inner.merge(other.inner)
        return VectorChoiceMap(new), VectorChoiceMap(discard)

    @dispatch
    def merge(self, other: EmptyChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return self, other

    ###########
    # Dunders #
    ###########

    def __hash__(self):
        return hash(self.inner)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = Tree("[bold](Vector)")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


#####
# IndexChoiceMap
#####


@dataclass
class IndexChoiceMap(ChoiceMap):
    indices: IntArray
    inner: ChoiceMap

    def flatten(self):
        return (self.indices, self.inner), ()

    @classmethod
    def convert(cls, chm: ChoiceMap) -> "IndexChoiceMap":
        indices = []
        subtrees = []
        for (k, v) in chm.get_subtrees_shallow():
            if isinstance(k, IntArray):
                indices.append(k)
                subtrees.append(v)
            else:
                raise Exception(
                    f"Failed to convert choice map of type {type(chm)} to IndexChoiceMap."
                )

        inner = tree_stack(subtrees)
        indices = jnp.array(indices)
        return IndexChoiceMap.new(inner, indices)

    @classmethod
    @dispatch
    def new(cls, indices: IntArray, inner: ChoiceMap) -> ChoiceMap:
        # Promote raw integers (or scalars) to non-null leading dim.
        indices = jnp.array(indices)
        if not indices.shape:
            indices = indices[:, None]

        # Verify that dimensions are consistent before creating an
        # `IndexChoiceMap`.
        _ = static_check_leaf_length((inner, indices))

        # if you try to wrap around an EmptyChoiceMap, do nothing.
        if isinstance(inner, EmptyChoiceMap):
            return inner

        return IndexChoiceMap(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: List, inner: ChoiceMap) -> ChoiceMap:
        indices = jnp.array(indices)
        return IndexChoiceMap.new(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: Any, inner: Dict) -> ChoiceMap:
        inner = choice_map(inner)
        return IndexChoiceMap.new(indices, inner)

    def is_empty(self):
        return self.inner.is_empty()

    def has_subtree(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))

    @dispatch
    def get_subtree(self, addr: Tuple):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return EmptyChoiceMap()
        (idx, addr) = addr
        subtree = self.inner.get_subtree(addr)
        if isinstance(subtree, EmptyChoiceMap):
            return EmptyChoiceMap()
        else:
            (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
            subtree = jtu.tree_map(lambda v: v[slice_index], subtree)
            return mask(idx in self.indices, subtree)

    @dispatch
    def get_subtree(self, idx: IntArray):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        subtree = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return mask(jnp.isin(idx, self.indices), subtree)

    def get_selection(self):
        inner_selection = self.inner.get_selection()
        return IndexSelection.new(self.indices, inner_selection)

    def get_subtrees_shallow(self):
        raise NotImplementedError

    def merge(self, _: ChoiceMap):
        raise Exception("TODO: can't merge IndexChoiceMaps")

    def get_index(self):
        return self.indices

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.indices, short_arrays=True)
        sub_tree = Tree(f"[bold](Index,{doc})")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


##############
# Shorthands #
##############

vector_choice_map = VectorChoiceMap.new
index_choice_map = IndexChoiceMap.new
index_select = IndexSelection.new
