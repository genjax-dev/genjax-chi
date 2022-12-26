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

from genjax._src.core.datatypes import AllSelection
from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import EmptyChoiceMap
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import NoneSelection
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import Trace
from genjax._src.core.datatypes import ValueChoiceMap
from genjax._src.core.tracetypes import TraceType
from genjax._src.core.tree import Leaf
from genjax._src.core.typing import FloatArray
from genjax._src.generative_functions.builtin.builtin_tracetype import (
    BuiltinTraceType,
)
from genjax._src.generative_functions.builtin.trie import Trie


#####
# ChoiceMap
#####


@dataclass
class BuiltinChoiceMap(ChoiceMap):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

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
        return BuiltinChoiceMap(trie)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return EmptyChoiceMap()
        else:
            return value

    def get_subtrees_shallow(self):
        return self.trie.get_subtrees_shallow()

    def get_selection(self):
        trie = Trie.new()
        for (k, v) in self.get_subtrees_shallow():
            trie[k] = v.get_selection()
        return BuiltinSelection(trie)

    def merge(self, other):
        trie = Trie.new()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other.get_subtree(k)
                trie[k] = v.merge(sub)
            else:
                trie[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                trie[k] = v
        return BuiltinChoiceMap(trie)

    def __setitem__(self, k, v):
        self.trie[k] = v

    def __getitem__(self, k):
        value = self.get_subtree(k)
        if isinstance(value, Leaf):
            return value.get_leaf_value()
        else:
            return value

    def __hash__(self):
        return hash(self.trie)


#####
# Selection
#####


@dataclass
class BuiltinSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @classmethod
    def new(cls, selected):
        assert isinstance(selected, list)
        trie = Trie.new()
        for k in selected:
            trie[k] = AllSelection()
        return BuiltinSelection(trie)

    def filter(self, chm):
        def _inner(k, v):
            if k in self.trie:
                sub = self.trie[k]
                under, s = sub.filter(v)
                return k, under, s
            else:
                return k, EmptyChoiceMap(), 0.0

        trie = Trie.new()
        score = 0.0
        iter = chm.get_subtrees_shallow()
        for (k, v, s) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v
                score += s

        if isinstance(chm, TraceType):
            return BuiltinTraceType(trie, chm.get_rettype()), score
        else:
            return BuiltinChoiceMap(trie), score

    def complement(self):
        return BuiltinComplementSelection(self.trie)

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

    def merge(self, other):
        trie = Trie.new()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other[k]
                trie[k] = v.merge(sub)
            else:
                trie[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                trie[k] = v
        return BuiltinSelection(trie)


@dataclass
class BuiltinComplementSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.selection,), ()

    def filter(self, chm):
        def _inner(k, v):
            if k in self.trie:
                sub = self.trie[k]
                v, s = sub.complement().filter(v)
                return k, v, s
            else:
                return k, v, 0.0

        trie = Trie.new()
        score = 0.0
        iter = chm.get_subtrees_shallow()
        for (k, v, s) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v
                score += s

        if isinstance(chm, TraceType):
            return type(chm)(trie, chm.get_rettype()), score
        else:
            return BuiltinChoiceMap(trie), score

    def complement(self):
        return BuiltinSelection(self.trie)

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

    def merge(self, other):
        trie = Trie.new()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other[k]
                trie[k] = v.merge(sub)
            else:
                trie[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                trie[k] = v
        return BuiltinComplementSelection(trie)


#####
# Trace
#####


@dataclass
class BuiltinTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Trie
    cache: Trie
    score: FloatArray

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
        return BuiltinChoiceMap(self.choices)

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
