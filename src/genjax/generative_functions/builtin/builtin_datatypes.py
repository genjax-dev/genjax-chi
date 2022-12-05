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
from typing import Any
from typing import Dict
from typing import Tuple

import jax.numpy as jnp
import rich

import genjax.core.pretty_printing as gpp
from genjax.core.datatypes import AllSelection
from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Selection
from genjax.core.datatypes import Trace
from genjax.core.datatypes import ValueChoiceMap
from genjax.core.hashabledict import HashableDict
from genjax.core.hashabledict import hashabledict
from genjax.core.tracetypes import TraceType
from genjax.core.tree import Tree
from genjax.generative_functions.builtin.builtin_tracetype import (
    BuiltinTraceType,
)


#####
# BuiltinTrie
#####


@dataclass
class BuiltinTrie(Tree):
    inner: HashableDict

    def flatten(self):
        return (self.inner,), ()

    def trie_insert(self, addr, value):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if first not in self.inner:
                subtree = BuiltinTrie(hashabledict())
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
                raise Exception(f"Tree has no subtree at {first}")
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
        return BuiltinTrie(new)

    def __setitem__(self, k, v):
        self.trie_insert(k, v)

    def __hash__(self):
        return hash(self.inner)

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        for (k, v) in self.inner.items():
            subk = tree.add(f"[bold green]{k}")
            if hasattr(v, "_build_rich_tree"):
                subtree = v._build_rich_tree()
                subk.add(subtree)
            else:
                subk.add(gpp.tree_pformat(v))
        return tree


#####
# ChoiceMap
#####


@dataclass
class BuiltinChoiceMap(ChoiceMap):
    inner: HashableDict

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    def new(cls, constraints):
        assert isinstance(constraints, Dict)
        fresh = BuiltinChoiceMap(hashabledict())
        for (k, v) in constraints.items():
            v = (
                ValueChoiceMap(v)
                if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
                else v
            )
            fresh.trie_insert(k, v)
        return fresh

    def trie_insert(self, addr, value):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if first not in self.inner:
                subtree = BuiltinChoiceMap(hashabledict())
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
                raise Exception(f"Tree has no subtree at {first}")
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            if addr not in self.inner:
                return EmptyChoiceMap()
            return self.inner[addr]

    def get_subtrees_shallow(self):
        return self.inner.items()

    def get_selection(self):
        new_tree = hashabledict()
        for (k, v) in self.get_subtrees_shallow():
            new_tree[k] = v.get_selection()
        return BuiltinSelection(new_tree)

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
        return BuiltinChoiceMap(new)

    def __setitem__(self, k, v):
        self.trie_insert(k, v)

    def __hash__(self):
        return hash(self.inner)


#####
# Trace
#####


@dataclass
class BuiltinTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: BuiltinChoiceMap
    cache: BuiltinTrie
    score: jnp.float32

    def flatten(self):
        return (
            self.args,
            self.retval,
            self.choices,
            self.cache,
            self.score,
        ), (self.gen_fn,)

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return self.choices

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def has_cached_value(self, addr):
        return self.cache.has_subtree(addr)

    def get_cached_value(self, addr):
        return self.cache.get_subtree(addr)


#####
# Selection
#####


@dataclass
class BuiltinSelection(Selection):
    inner: HashableDict

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    def new(cls, selected):
        assert isinstance(selected, list)
        inner = hashabledict()
        new = BuiltinSelection(inner)
        for k in selected:
            new.trie_insert(k, AllSelection())
        return new

    def trie_insert(self, addr, value):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if first not in self.inner:
                subtree = BuiltinSelection(hashabledict())
                self.inner[first] = subtree
            subtree = self.inner[first]
            subtree.trie_insert(rest, value)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            self.inner[addr] = value

    def filter(self, chm):
        def _inner(k, v):
            if k in self.inner:
                sub = self.inner[k]
                under, s = sub.filter(v)
                return k, under, s
            else:
                return k, EmptyChoiceMap(), 0.0

        new_tree = hashabledict()
        score = 0.0
        iter = chm.get_subtrees_shallow()
        for (k, v, s) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                new_tree[k] = v
                score += s

        if isinstance(chm, TraceType):
            return BuiltinTraceType(new_tree, chm.get_rettype()), score
        else:
            return BuiltinChoiceMap(new_tree), score

    def complement(self):
        return BuiltinComplementSelection(self.inner)

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
                raise Exception(f"Tree has no subtree at {first}")
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return self.inner[addr]

    def get_subtrees_shallow(self):
        return self.inner.items()

    def merge(self, other):
        new = hashabledict()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other[k]
                new[k] = v.merge(sub)
            else:
                new[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                new[k] = v
        return BuiltinSelection(new)


@dataclass
class BuiltinComplementSelection(Selection):
    inner: HashableDict

    def flatten(self):
        return (self.inner,), ()

    def filter(self, chm):
        def _inner(k, v):
            if k in self.inner:
                sub = self.inner[k]
                v, s = sub.complement().filter(v)
                return k, v, s
            else:
                return k, v, 0.0

        new_tree = hashabledict()
        score = 0.0
        iter = chm.get_subtrees_shallow()
        for (k, v, s) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                new_tree[k] = v
                score += s

        if isinstance(chm, TraceType):
            return type(chm)(new_tree, chm.get_rettype()), score
        else:
            return BuiltinChoiceMap(new_tree), score

    def complement(self):
        new_tree = dict()
        for (k, v) in self.inner.items():
            new_tree[k] = v.complement()
        return BuiltinSelection(new_tree)

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
                raise Exception(f"Tree has no subtree at {first}")
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return self.inner[addr]

    def get_subtrees_shallow(self):
        return self.inner.items()

    def merge(self, other):
        new = hashabledict()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other[k]
                new[k] = v.merge(sub)
            else:
                new[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                new[k] = v
        return BuiltinComplementSelection(new)
