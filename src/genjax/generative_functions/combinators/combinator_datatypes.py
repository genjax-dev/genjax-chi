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
from rich.tree import Tree

import genjax.core.pretty_printing as gpp
from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import Selection
from genjax.core.datatypes import Trace
from genjax.core.masks import BooleanMask
from genjax.core.typing import Integer
from genjax.core.typing import IntegerTensor


######################################
# Vector-shaped combinator datatypes #
######################################

# This section applies to `Map` and `Unfold`, currently.

#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    indices: IntegerTensor
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.indices, self.inner), ()

    def get_selection(self):
        subselection = self.inner.get_selection()
        return VectorSelection.new(self.indices, subselection)

    @classmethod
    def new(cls, indices, inner):
        if isinstance(inner, EmptyChoiceMap):
            return inner
        return VectorChoiceMap(indices, inner)

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return VectorChoiceMap.new(
            self.indices,
            self.inner.get_subtree(addr),
        )

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return k, VectorChoiceMap.new(self.indices, v)

        return map(
            lambda args: _inner(*args),
            self.inner.get_subtrees_shallow(),
        )

    def merge(self, other):
        return VectorChoiceMap.new(self.indices, self.inner.merge(other))

    def __hash__(self):
        return hash(self.inner)

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
    index: Integer
    submaps: Sequence[Union[ChoiceMap, Trace]]

    def flatten(self):
        return (self.index, self.submaps), ()

    def has_subtree(self, addr):
        checks = list(map(lambda v: v.has_subtree(addr), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    def get_subtree(self, addr):
        submaps = list(map(lambda v: v.get_subtree(addr), self.submaps))
        return IndexedChoiceMap.new(self.index, submaps)

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
        for subt in subts:
            tree.add(subt)
        return tree


#####
# IndexedSelection
#####


@dataclass
class IndexedSelection(Selection):
    index: IntegerTensor
    subselections: Sequence[Selection]
