# Copyright 2024 MIT Probabilistic Computing Project
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


import jax.numpy as jnp
import jax.tree_util as jtu
import rich.tree as rich_tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import (
    ChoiceMap,
    EmptyChoice,
    MapSelection,
    Mask,
    Selection,
    TraceSlice,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Int,
    IntArray,
    Tuple,
    dispatch,
    static_check_is_concrete,
)

######################################
# Vector-shaped combinator datatypes #
######################################

# The data types in this section are used in `Map` and `Unfold`, currently.

#####################
# Indexed datatypes #
#####################


class IndexedSelection(MapSelection):
    indices: IntArray
    inner: Selection

    def __post_init__(self):
        Pytree.static_check_tree_leaves_have_matching_leading_dim(
            (self.inner, self.indices)
        )

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


class IndexedChoiceMap(ChoiceMap):
    indices: IntArray
    inner: ChoiceMap

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter_selection(
        self,
        selection: MapSelection,
    ) -> ChoiceMap:
        return IndexedChoiceMap(self.indices, self.inner.filter_selection(selection))

    @dispatch
    def filter_selection(
        self,
        selection: IndexedSelection,
    ) -> ChoiceMap:
        flags = jnp.isin(selection.indices, self.indices)
        filtered_inner = self.inner.filter(selection.inner)
        masked = Mask(flags, filtered_inner)
        return IndexedChoiceMap(self.indices, masked)

    @dispatch
    def has_submap(self, addr: IntArray):
        return addr in self.indices

    @dispatch
    def has_submap(self, addr: Tuple):
        return jnp.logical_and(addr[0] in self.indices, self.inner.has_submap(addr[1:]))

    # where we left off: get slice working here
    # see if we can disentangle and-or de-@dispatch these things

    @dispatch
    def get_submap(self, addr: Tuple):
        if len(addr) == 1:
            return self.get_submap(addr[0])
        idx = addr[0]
        (slice_index,) = jnp.nonzero(jnp.atleast_1d(idx == self.indices), size=1)
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        submap = submap.get_submap(addr[1:])
        if isinstance(submap, EmptyChoice):
            return submap
        else:
            return Mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, idx: Int):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = self.indices[slice_index[0]] if self.indices.shape else idx
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return Mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, idx: IntArray):
        (slice_index,) = jnp.nonzero(jnp.atleast_1d(idx == self.indices), size=1)
        slice_index = self.indices[slice_index[0]] if self.indices.shape else idx

        # TODO(colin): Ask McCoy: what does the line below do? Without it, this and the upper implementation
        # would be the same. I can see that this is the route taken by JAX tracing, but why would inner need
        # an array wrapping?

        # inner = jtu.tree_map(lambda v: jnp.array(v, copy=False), self.inner)
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return Mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, _: Any):
        return EmptyChoice()

    def get_selection(self):
        return self.inner.get_selection()

    # TODO: this will fail silently if the indices of the incoming map
    # are different than the original map.
    @dispatch
    def merge(self, new: "IndexedChoiceMap"):
        new_inner, discard = self.inner.merge(new.inner)
        assert discard.is_empty()
        return IndexedChoiceMap(self.indices, new_inner)

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


class VectorChoiceMap(ChoiceMap):
    inner: Any

    def __post_int__(self):
        Pytree.static_check_tree_leaves_have_matching_leading_dim(self.inner)

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter_selection(
        self,
        selection: IndexedSelection,
    ) -> ChoiceMap:
        inner = self.inner.filter(selection.inner)
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(inner)
        check = selection.indices <= dim
        idxs = check * selection.indices
        return IndexedChoiceMap(
            selection.indices, jtu.tree_map(lambda v: v[idxs], inner)
        )

    # where we left off: let's  do this with a slice
    # BUT expand out to a list of indices for the IndexedChoiceMap.
    # that's the easiest way to make everyone happy.
    @dispatch
    def filter_selection(
        self,
        selection: Selection,
    ) -> ChoiceMap:
        if not isinstance(selection, TraceSlice):
            return VectorChoiceMap(self.inner.filter(selection))
        if selection.s == () or selection.s[0] == slice(None, None, None):
            return self
        inner = self.inner.filter_selection(selection.inner)
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(inner)
        # We allow bare integers, jnp arrays of integers, and start:stop:stride
        # slices as individual members of a TraceSlice. We convert each of these
        # to a jnp array and use that to construct an IndexedChoiceMap.
        indices = selection.indices
        if isinstance(indices, slice):
            # TODO(colin): consider making it so that an IndexedChoiceMap can contain a slice directly,
            # instead of forcing us to convert to a jnp.array
            indices = jnp.array(range(*indices.indices(dim)))

        return IndexedChoiceMap(indices, jtu.tree_map(lambda v: v[indices], inner))

    def get_selection(self):
        subselection = self.inner.get_selection()
        # Static: get the leading dimension size value.
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(
            self.inner,
        )
        return IndexedSelection(jnp.arange(dim), subselection)

    @dispatch
    def has_submap(self, addr: IntArray):
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(
            self.inner,
        )
        return addr < dim

    @dispatch
    def has_submap(self, addr: Tuple):
        (idx, *addr) = addr
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(
            self.inner,
        )
        return jnp.logical_and(idx < dim, self.inner.has_submap(tuple(addr)))

    @dispatch
    def get_submap(self, slc: slice):
        sliced = jtu.tree_map(lambda v: v[slc], self.inner)
        return sliced

    @dispatch
    def get_submap(self, idx: Int):
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(
            self.inner,
        )
        check = idx < dim
        idx = check * idx
        sliced = jtu.tree_map(lambda v: v[idx], self.inner)
        return sliced

    @dispatch
    def get_submap(self, idx: IntArray):
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(
            self.inner,
        )
        check = idx < dim
        idx = check * idx
        sliced = jtu.tree_map(lambda v: v[idx], self.inner)
        if static_check_is_concrete(check) and check:
            return sliced
        else:
            return Mask(idx < dim, sliced)

    @dispatch
    def get_submap(self, addr: Tuple):
        (idx, *addr) = addr
        sliced = self.get_submap(idx)
        sliced = sliced.get_submap(tuple(addr))
        return sliced

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

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](VectorChoiceMap)")
        tree.add(self.inner.__rich_tree__())
        return tree
