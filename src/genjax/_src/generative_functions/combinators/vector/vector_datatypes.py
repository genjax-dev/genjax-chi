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
from jax import vmap

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import (
    ChoiceMap,
    ChoiceValue,
    EmptyChoice,
    HierarchicalChoiceMap,
    Mask,
    Selection,
)
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Dict,
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


class IndexedChoiceMap(ChoiceMap):
    indices: IntArray
    inner: ChoiceMap

    @classmethod
    def from_dict(cls, d: Dict[int, Any]) -> ChoiceMap:
        """Produce an IndexedChoiceMap from a dictionary with integer keys.

        IndexedChoiceMap.from_dict({
          1: 1.0,
          2: 3.0
        })

        is equivalent to indexed_choice_map([1, 2], choice_map({"x": [1.0, 3.0]}))
        """
        sorted_keys = sorted(d.keys())
        td = dict()
        td["x"] = ChoiceValue(jnp.array([d[k] for k in sorted_keys]))
        return IndexedChoiceMap(jnp.array(sorted_keys), HierarchicalChoiceMap(Trie(td)))

    def is_empty(self):
        return self.inner.is_empty()

    def filter(
        self,
        selection: Selection,
    ) -> ChoiceMap:
        def _filter(idx, inner):
            check, remaining = selection.both(idx)
            return IndexedChoiceMap(
                idx,
                Mask(check, inner.filter(remaining)),
            )

        return vmap(_filter)(self.indices, self.inner)

    @dispatch
    def has_submap(self, addr: IntArray):
        return addr in self.indices

    @dispatch
    def has_submap(self, addr: Tuple):
        (idx, *addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_submap(tuple(addr)))

    @dispatch
    def get_submap(self, addr: Tuple):
        if len(addr) == 1:
            return self.get_submap(addr[0])
        idx = addr[0]
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
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
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = self.indices[slice_index[0]] if self.indices.shape else idx
        inner = jtu.tree_map(lambda v: jnp.array(v, copy=False), self.inner)
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, inner)
        return Mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, _: Any):
        return EmptyChoice()

    def get_selection(self):
        subselection = self.inner.get_selection()
        return vmap(Selection.idx)(self.indices) > subselection

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

    def __post_init__(self):
        Pytree.static_check_tree_leaves_have_matching_leading_dim(self.inner)

    def is_empty(self):
        return self.inner.is_empty()

    def filter(
        self,
        selection: Selection,
    ) -> ChoiceMap:
        def _filter(idx, inner):
            check, remaining = selection.both(idx)
            return VectorChoiceMap(
                Mask(check, inner.filter(remaining)),
            )

        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(self.inner)
        idxs = jnp.arange(dim)
        return vmap(_filter)(idxs, self.inner)

    def get_selection(self):
        subselection = self.inner.get_selection()
        dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(self.inner)
        idxs = jnp.arange(dim)
        return vmap(Selection.idx)(idxs) > subselection

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
