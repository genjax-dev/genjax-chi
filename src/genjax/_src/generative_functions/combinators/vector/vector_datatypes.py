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
import rich.tree as rich_tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoice
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import choice_map
from genjax._src.core.datatypes.generative import mask
from genjax._src.core.datatypes.generative import select
from genjax._src.core.pytree.checks import (
    static_check_tree_leaves_have_matching_leading_dim,
)
from genjax._src.core.pytree.checks import static_check_tree_structure_equivalence
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch


######################################
# Vector-shaped combinator datatypes #
######################################

# The data types in this section are used in `Map` and `Unfold`, currently.

#####################
# Indexed datatypes #
#####################


@dataclass
class IndexedSelection(Selection):
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
        return IndexedSelection(idxs, AllSelection())

    @classmethod
    @dispatch
    def new(cls, idx: Any, inner: Selection):
        idxs = jnp.array(idx)
        return IndexedSelection(idxs, inner)

    @classmethod
    @dispatch
    def new(cls, idx: Any, *inner: Any):
        idxs = jnp.array(idx)
        inner = select(*inner)
        return IndexedSelection(idxs, inner)

    @dispatch
    def has_addr(self, addr: IntArray):
        return jnp.isin(addr, self.indices)

    @dispatch
    def has_addr(self, addr: Tuple):
        if len(addr) <= 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_addr(addr))

    def get_subselection(self, addr):
        return self.index_selection.get_subselection(addr)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        doc = gpp._pformat_array(self.indices, short_arrays=True)
        tree = rich_tree.Tree(f"[bold](IndexedSelection, {doc})")
        tree.add(self.inner.__rich_tree__())
        return tree


@dataclass
class IndexedChoiceMap(ChoiceMap):
    indices: IntArray
    inner: ChoiceMap

    def flatten(self):
        return (self.indices, self.inner), ()

    @classmethod
    @dispatch
    def new(cls, indices: IntArray, inner: ChoiceMap) -> ChoiceMap:
        # Promote raw integers (or scalars) to non-null leading dim.
        indices = jnp.array(indices, copy=False)

        # Verify that dimensions are consistent before creating an
        # `IndexedChoiceMap`.
        _ = static_check_tree_leaves_have_matching_leading_dim((inner, indices))

        # if you try to wrap around an EmptyChoice, do nothing.
        if isinstance(inner, EmptyChoice):
            return inner

        return IndexedChoiceMap(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: List, inner: ChoiceMap) -> ChoiceMap:
        indices = jnp.array(indices)
        return IndexedChoiceMap.new(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: Any, inner: Dict) -> ChoiceMap:
        inner = choice_map(inner)
        return IndexedChoiceMap.new(indices, inner)

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        return IndexedChoiceMap(self.indices, self.inner.filter(selection))

    def has_submap(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_submap(addr))

    @dispatch
    def filter(
        self,
        selection: IndexedSelection,
    ) -> ChoiceMap:
        flags = jnp.isin(selection.indices, self.indices)
        filtered_inner = self.inner.filter(selection.inner)
        masked = mask(flags, filtered_inner)
        return IndexedChoiceMap(self.indices, masked)

    def has_submap(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_submap(addr))

    @dispatch
    def get_submap(self, addr: Tuple):
        if len(addr) == 1:
            return self.get_submap(addr[0])
        idx, *rest = addr
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        submap = mask(jnp.isin(idx, self.indices), submap)
        return submap.get_submap(tuple(rest))

    @dispatch
    def get_submap(self, idx: IntArray):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, idx: Int):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, _: Any):
        return EmptyChoice()

    def get_selection(self):
        return self.inner.get_selection()

    def merge(self, _: ChoiceMap):
        raise Exception("TODO: can't merge IndexedChoiceMaps")

    def get_index(self):
        return self.indices

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        doc = gpp._pformat_array(self.indices, short_arrays=True)
        tree = rich_tree.Tree(f"[bold](IndexedChoiceMap, {doc})")
        sub_tree = self.inner.__rich_tree__()
        tree.add(sub_tree)
        return tree


#####################
# Vector choice map #
#####################


@dataclass
class VectorChoiceMap(ChoiceMap):
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    @dispatch
    def new(
        cls,
        inner: EmptyChoice,
    ) -> EmptyChoice:
        return inner

    @classmethod
    @dispatch
    def new(
        cls,
        inner: Choice,
    ) -> ChoiceMap:
        # Static assertion: all leaves must have same first dim size.
        static_check_tree_leaves_have_matching_leading_dim(inner)
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
        selection: IndexedSelection,
    ) -> ChoiceMap:
        inner = self.inner.filter(selection.inner)
        return IndexedChoiceMap(selection.indices, inner)

    @dispatch
    def filter(
        self,
        selection: Selection,
    ) -> ChoiceMap:
        return VectorChoiceMap.new(self.inner.filter(selection))

    def get_selection(self):
        subselection = self.inner.get_selection()
        # Static: get the leading dimension size value.
        dim = static_check_tree_leaves_have_matching_leading_dim(
            self.inner,
        )
        return IndexedSelection(jnp.arange(dim), subselection)

    def has_submap(self, addr):
        return self.inner.has_submap(addr)

    def get_submap(self, addr):
        return self.inner.get_submap(addr)

    @dispatch
    def merge(self, other: "VectorChoiceMap") -> Tuple[ChoiceMap, ChoiceMap]:
        new, discard = self.inner.merge(other.inner)
        return VectorChoiceMap(new), VectorChoiceMap(discard)

    @dispatch
    def merge(self, other: IndexedChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        indices = other.indices

        sliced = jtu.tree_map(lambda v: v[indices], self.inner)
        new, discard = sliced.merge(other.inner)

        def _inner(v1, v2):
            return v1.at[indices].set(v2)

        assert jtu.tree_structure(self.inner) == jtu.tree_structure(new)
        new = jtu.tree_map(_inner, self.inner, new)

        return VectorChoiceMap(new), IndexedChoiceMap(indices, discard)

    @dispatch
    def merge(self, other: EmptyChoice) -> Tuple[ChoiceMap, ChoiceMap]:
        return self, other

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich_tree.Tree("[bold](Vector)")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


################################
# Extend select and choice_map #
################################


@dispatch
def select(idx: Union[Int, IntArray], *args):
    return indexed_select(idx, *args)


@dispatch
def choice_map(indices: List[Int], submaps: List[ChoiceMap]):
    # submaps must have same Pytree structure to use
    # optimized representation.
    assert static_check_tree_structure_equivalence(submaps)
    index_arr = jnp.array(indices)
    return indexed_choice_map(index_arr, submaps)


@dispatch
def choice_map(index: Int, submap: ChoiceMap):
    expanded = jtu.tree_map(lambda v: jnp.expand_dims(v, axis=0), submap)
    return indexed_choice_map([index], expanded)


##############
# Shorthands #
##############

indexed_choice_map = IndexedChoiceMap.new
indexed_select = IndexedSelection.new
vector_choice_map = VectorChoiceMap.new
