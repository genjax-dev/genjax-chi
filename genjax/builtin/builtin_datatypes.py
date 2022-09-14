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

import jax.numpy as jnp
from dataclasses import dataclass
from genjax.core.datatypes import (
    ValueChoiceMap,
    ChoiceMap,
    Trace,
    GenerativeFunction,
)
from typing import Any, Tuple
from genjax.core.hashabledict import HashableDict, hashabledict
from genjax.core.datatypes import EmptyChoiceMap, Selection, AllSelection
from genjax.core.tracetypes import TraceType
from typing import Dict

#####
# ChoiceMap
#####


@dataclass
class BuiltinChoiceMap(ChoiceMap):
    inner: HashableDict

    def __init__(self, constraints):
        if isinstance(constraints, BuiltinChoiceMap):
            self.inner = constraints.inner
        else:
            self.inner = constraints

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

    def has_value(self):
        return False

    def get_value(self):
        raise Exception("BuiltinChoiceMap is not a value choice map.")

    def has_choice(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_choice(first):
                subtree = self.get_choice(first)
                return subtree.has_choice(rest)
            else:
                return False
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return addr in self.inner

    def get_choice(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_choice(first):
                subtree = self.get_choice(first)
                return subtree.get_choice(rest)
            else:
                raise Exception(f"Tree has no subtree at {first}")
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return self.inner[addr]

    def get_choices_shallow(self):
        return self.inner.items()

    def get_selection(self):
        new_tree = hashabledict()
        for (k, v) in self.get_choices_shallow():
            new_tree[k] = v.get_selection()
        return BuiltinSelection(new_tree)

    def merge(self, other):
        new = hashabledict()
        for (k, v) in self.get_choices_shallow():
            if other.has_choice(k):
                sub = other[k]
                new[k] = v.merge(sub)
            else:
                new[k] = v
        for (k, v) in other.get_choices_shallow():
            if not self.has_choice(k):
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
    score: jnp.float32

    def flatten(self):
        return (self.args, self.retval, self.choices, self.score), (
            self.gen_fn,
        )

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


#####
# Selection
#####


@dataclass
class BuiltinSelection(Selection):
    inner: HashableDict

    def __init__(self, selected):
        self.inner = hashabledict()
        if isinstance(selected, list):
            for k in selected:
                self.inner[k] = AllSelection()
        if isinstance(selected, Dict):
            self.inner = HashableDict(selected)

    def flatten(self):
        return (self.inner,), ()

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
        iter = (
            chm.get_types_shallow()
            if isinstance(chm, TraceType)
            else chm.get_choices_shallow()
        )
        for (k, v, s) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                new_tree[k] = v
                score += s
        if isinstance(chm, TraceType):
            return type(chm)(new_tree, chm.get_rettype()), score
        else:
            return BuiltinChoiceMap(new_tree), score

    def complement(self):
        return BuiltinComplementSelection(self.inner)


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
        iter = (
            chm.get_types_shallow()
            if isinstance(chm, TraceType)
            else chm.get_choices_shallow()
        )
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
