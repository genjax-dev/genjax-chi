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

from genjax._src.core.datatypes import AllSelection
from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import EmptyChoiceMap
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import NoneSelection
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import Trace
from genjax._src.core.datatypes import ValueChoiceMap
from genjax._src.core.tracetypes import TraceType
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
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
        return BuiltinChoiceMap(trie)

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
            if not trie.has_subtree(k):
                trie[k] = v
        return BuiltinChoiceMap(trie)

    def __setitem__(self, k, v):
        self.trie[k] = v

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

    @typecheck
    @classmethod
    def new(cls, selected: List):
        trie = Trie.new()
        for k in selected:
            trie[k] = AllSelection()
        return BuiltinSelection(trie)

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
            return BuiltinTraceType(trie, tree.get_rettype())
        else:
            return BuiltinChoiceMap(trie)

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
            sub = self.trie[k]
            if sub is None:
                sub = AllSelection()
            under = sub.filter(v)
            return k, under

        trie = Trie.new()
        iter = chm.get_subtrees_shallow()
        for (k, v) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        if isinstance(chm, TraceType):
            return type(chm)(trie, chm.get_rettype())
        else:
            return BuiltinChoiceMap(trie)

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
            self.gen_fn,
            self.args,
            self.retval,
            self.choices,
            self.cache,
            self.score,
        ), ()

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

    def project(self, selection: Selection):
        weight = 0.0
        for (k, v) in self.choices.get_subtrees_shallow():
            if selection.has_subtree(k):
                weight += v.project(selection[k])
        return weight

    def has_cached_value(self, addr):
        return self.cache.has_subtree(addr)

    def get_cached_value(self, addr):
        return self.cache.get_subtree(addr)
