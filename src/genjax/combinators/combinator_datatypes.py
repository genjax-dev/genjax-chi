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

import genjax.core.pretty_printing as gpp
from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import Trace
from genjax.core.masks import BooleanMask
from genjax.core.specialization import is_concrete


Int32 = Union[jnp.int32, np.int32]
IntTensor = Union[jnp.ndarray, np.ndarray]

#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    indices: IntTensor
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.indices, self.inner), ()

    @classmethod
    def new(cls, indices, inner):
        if isinstance(inner, EmptyChoiceMap):
            return inner
        return VectorChoiceMap(indices, inner)

    def is_leaf(self):
        return self.inner.is_leaf()

    def get_leaf_value(self):
        return self.inner.get_leaf_value()

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return VectorChoiceMap.new(self.indices, self.inner.get_subtree(addr))

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return k, VectorChoiceMap.new(self.indices, v)

        return map(
            lambda args: _inner(*args), self.inner.get_subtrees_shallow()
        )

    def merge(self, other):
        return VectorChoiceMap.new(self.indices, self.inner.merge(other))

    def get_score(self):
        return self.inner.get_score()

    def get_index(self):
        return self.indices

    def get_selection(self):
        return self.inner.get_selection()

    def __hash__(self):
        return hash(self.inner)

    def tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        subt = self.inner.build_rich_tree()
        subk = Tree("[blue]indices")
        subk.add(gpp.tree_pformat(self.indices))
        tree.add(subk)
        tree.add(subt)
        return tree


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
    index: Int32
    submaps: Sequence[Union[ChoiceMap, Trace]]

    def flatten(self):
        return (self.index, self.submaps), ()

    def is_leaf(self):
        checks = list(map(lambda v: v.is_leaf(), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    def get_leaf_value(self):
        leafs = list(
            map(
                lambda v: jnp.array(False)
                if (not v.is_leaf()) or isinstance(v, EmptyChoiceMap)
                else v.get_leaf_value(),
                self.submaps,
            )
        )
        return jnp.choose(self.index, leafs, mode="wrap")

    def has_subtree(self, addr):
        checks = list(map(lambda v: v.has_subtree(addr), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    def get_subtree(self, addr):
        submaps = list(map(lambda v: v.get_subtree(addr), self.submaps))
        return IndexedChoiceMap(self.index, submaps)

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

    def merge(self, other):
        new_submaps = list(map(lambda v: v.merge(other), self.submaps))
        return IndexedChoiceMap(self.index, new_submaps)

    @classmethod
    def collapse(cls, v):
        def _inner(v):
            if isinstance(v, IndexedChoiceMap) and is_concrete(v.index):
                return IndexedChoiceMap.collapse(v.submaps[v.index])
            else:
                return v

        def _check(v):
            return isinstance(v, IndexedChoiceMap)

        return jtu.tree_map(_inner, v, is_leaf=_check)

    @classmethod
    def collapse_boundary(cls, fn):
        def _inner(self, key, *args, **kwargs):
            args = IndexedChoiceMap.collapse(args)
            return fn(self, key, *args, **kwargs)

        return _inner

    def __setitem__(self, k, v):
        for sub in self.submaps:
            sub[k] = v

    def tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        subts = list(map(lambda v: v.build_rich_tree(), self.submaps))
        tree.add(subts)
        return tree
