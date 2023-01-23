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

import itertools
from dataclasses import dataclass
from typing import Sequence
from typing import Union

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import EmptyChoiceMap
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import Trace
from genjax._src.core.masks import BooleanMask
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


###############################
# Switch combinator datatypes #
###############################

#####
# IndexedChoiceMap
#####

# Note that the abstract/concrete semantics of `jnp.choose`
# are slightly interesting. If we know ahead of time that
# the index is concrete, we can use `jnp.choose` without a
# fallback mode (e.g. index is out of bounds).
#
# If we do not know the index array ahead of time, we must
# choose a fallback mode to allow tracer values.


@dataclass
class IndexedChoiceMap(ChoiceMap):
    index: IntArray
    submaps: Sequence[Union[ChoiceMap, Trace]]

    def flatten(self):
        return (self.index, self.submaps), ()

    def has_subtree(self, addr):
        checks = list(map(lambda v: v.has_subtree(addr), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    # The way this works is slightly complicated, and relies on specific
    # assumptions about how SwitchCombinator works (and the
    # allowed shapes) of choice maps produced by SwitchCombinator.
    #
    # The key observation is that, if a branch choice map has an addr,
    # and it shares that address with another branch, the shape of the
    # choice map for each shared address has to be the same, all the
    # way down to the arguments.
    def get_subtree(self, addr):
        submaps = list(map(lambda v: v.get_subtree(addr), self.submaps))

        # Here, we create an index map before we filter out
        # EmptyChoiceMap instances.
        counter = 0
        index_map = []
        for v in submaps:
            if isinstance(v, EmptyChoiceMap):
                index_map.append(-1)
            else:
                index_map.append(counter)
                counter += 1
        index_map = jnp.array(index_map)

        non_empty_submaps = list(
            filter(lambda v: not isinstance(v, EmptyChoiceMap), submaps)
        )
        return jtu.tree_map(
            lambda *v: jnp.choose(
                index_map[self.index],
                v,
                mode="wrap",
            ),
            *non_empty_submaps,
        )

    def get_subtrees_shallow(self):
        def _inner(index, submap):
            check = index == self.index
            return map(
                lambda v: (v[0], BooleanMask.new(check, v[1])),
                submap.get_subtrees_shallow(),
            )

        sub_iterators = map(
            lambda args: _inner(*args),
            enumerate(self.submaps),
        )
        return itertools.chain(*sub_iterators)

    def get_selection(self):
        subselections = list(map(lambda v: v.get_selection(), self.submaps))
        return IndexedSelection.new(self.index, subselections)

    def merge(self, other):
        new_submaps = list(map(lambda v: v.merge(other), self.submaps))
        return IndexedChoiceMap.new(self.index, new_submaps)

    def _tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        subts = list(map(lambda v: v._build_rich_tree(), self.submaps))
        subk = Tree("[blue]index")
        subk.add(gpp.tree_pformat(self.index))
        tree.add(subk)
        for subt in subts:
            tree.add(subt)
        return tree


#####
# IndexedSelection
#####


@dataclass
class IndexedSelection(Selection):
    index: IntArray
    subselections: Sequence[Selection]
