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

import rich

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import NoneSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.datatypes.hashabledict import HashableDict
from genjax._src.core.datatypes.hashabledict import hashabledict
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.pretty_printing import CustomPretty
from genjax._src.core.typing import Dict
from genjax._src.core.typing import typecheck


#####
# Trie
#####


@dataclass
class Trie(ChoiceMap, CustomPretty):
    inner: HashableDict

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    def new(cls):
        return Trie(hashabledict())

    def get_selection(self):
        raise Exception("Trie doesn't provide conversion to Selection.")

    def get_choices(self):
        return TrieChoiceMap(self)

    def trie_insert(self, addr, value):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if first not in self.inner:
                subtree = Trie(hashabledict())
                self.inner[first] = subtree
            subtree = self.inner[first]
            subtree.trie_insert(rest, value)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            self.inner[addr] = value

    def has_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.has_subtree(rest)
            else:
                return False
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return addr in self.inner

    def get_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.get_subtree(rest)
            else:
                return None
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            if addr not in self.inner:
                return None
            return self.inner[addr]

    def get_subtrees_shallow(self):
        return self.inner.items()

    def merge(self, other):
        new = hashabledict()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other.get_subtree(k)
                new[k] = v.merge(sub)
            else:
                new[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                new[k] = v
        return Trie(new)

    def __setitem__(self, k, v):
        self.trie_insert(k, v)

    def __getitem__(self, k):
        return self.get_subtree(k)

    def __contains__(self, k):
        return self.has_subtree(k)

    def __hash__(self):
        return hash(self.inner)

    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        for (k, v) in self.inner.items():
            subk = tree.add(f"[bold]:{k}")
            subtree = gpp._pformat(v, **kwargs)
            subk.add(subtree)
        return tree


#####
# Trie-backed choice map
#####


@dataclass
class TrieChoiceMap(ChoiceMap):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @typecheck
    @classmethod
    def new(cls, constraints: Dict):
        assert isinstance(constraints, Dict)
        trie = Trie.new()
        for (k, v) in constraints.items():
            v = (
                ValueChoiceMap(v)
                if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
                else v
            )
            trie.trie_insert(k, v)
        return TrieChoiceMap(trie)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return EmptyChoiceMap()
        else:
            return value.get_choices()

    def get_subtrees_shallow(self):
        return map(
            lambda v: (v[0], v[1].get_choices()),
            self.trie.get_subtrees_shallow(),
        )

    def get_selection(self):
        trie = Trie.new()
        for (k, v) in self.get_subtrees_shallow():
            trie[k] = v.get_selection()
        return TrieSelection(trie)

    def __setitem__(self, k, v):
        self.trie[k] = v

    def __hash__(self):
        return hash(self.trie)


#####
# Trie-backed selection
#####


@dataclass
class TrieSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @typecheck
    @classmethod
    def new(cls, *addrs):
        trie = Trie.new()
        for addr in addrs:
            trie[addr] = AllSelection()
        return TrieSelection(trie)

    def filter(self, tree):
        def _inner(k, v):
            sub = self.trie[k]
            if sub is None:
                sub = NoneSelection()
            under = sub.filter(v)
            return k, under

        trie = Trie.new()
        iter = tree.get_subtrees_shallow()
        for (k, v) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        if isinstance(tree, TraceType):
            return type(tree)(trie, tree.get_rettype())
        else:
            return TrieChoiceMap(trie)

    def complement(self):
        return TrieComplementSelection(self.trie)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return NoneSelection()
        else:
            return value

    def get_subtrees_shallow(self):
        return self.trie.get_subtrees_shallow()


@dataclass
class TrieComplementSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.selection,), ()

    def filter(self, chm):
        def _inner(k, v):
            sub = self.trie[k]
            if sub is None:
                sub = NoneSelection()
            under = sub.complement().filter(v)
            return k, under

        trie = Trie.new()
        iter = chm.get_subtrees_shallow()
        for (k, v) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        if isinstance(chm, TraceType):
            return type(chm)(trie, chm.get_rettype())
        else:
            return TrieChoiceMap(trie)

    def complement(self):
        return TrieSelection(self.trie)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return NoneSelection()
        else:
            return value

    def get_subtrees_shallow(self):
        return self.trie.get_subtrees_shallow()


###################
# TrieConvertable #
###################

# A mixin: denotes that a choice map can be converted to a TrieChoiceMap


@dataclass
class TrieConvertable:
    def convert(self) -> TrieChoiceMap:
        new = TrieChoiceMap.new()
        for (k, v) in self.get_submaps_shallow():
            pass
        return new


##############
# Shorthands #
##############

choice_map = TrieChoiceMap.new
chm = choice_map
select = TrieSelection.new
sel = select
