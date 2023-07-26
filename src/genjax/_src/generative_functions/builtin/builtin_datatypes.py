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

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import NoneSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


##############
# Selections #
##############


@dataclass
class BuiltinSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @typecheck
    @classmethod
    def new(cls, *addrs):
        trie = Trie.new()
        for addr in addrs:
            trie[addr] = AllSelection()
        return BuiltinSelection(trie)

    @typecheck
    @classmethod
    def with_selections(cls, selections: Dict):
        assert isinstance(selections, Dict)
        trie = Trie.new()
        for (k, v) in selections.items():
            assert isinstance(v, Selection)
            trie.trie_insert(k, v)
        return BuiltinSelection(trie)

    def filter(self, tree):
        def _inner(k, v):
            sub = self.trie[k]
            if sub is None:
                sub = NoneSelection()

            # Handles hierarchical in Trie.
            elif isinstance(sub, Trie):
                sub = BuiltinSelection(sub)

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


@dataclass
class BuiltinComplementSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.selection,), ()

    def filter(self, chm):
        def _inner(k, v):
            sub = self.trie[k]
            if sub is None:
                sub = NoneSelection()

            # Handles hierarchical in Trie.
            elif isinstance(sub, Trie):
                sub = BuiltinSelection(sub)

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

    @dispatch
    def project(self, selection: AllSelection):
        return self.get_score()

    @dispatch
    def project(self, selection: NoneSelection):
        return 0.0

    @dispatch
    def project(self, selection: BuiltinSelection):
        weight = 0.0
        for (k, subtrace) in self.choices.get_subtrees_shallow():
            if selection.has_subtree(k):
                weight += subtrace.project(selection.get_subtree(k))
        return weight

    def has_cached_value(self, addr):
        return self.cache.has_subtree(addr)

    def get_cached_value(self, addr):
        return self.cache.get_subtree(addr)


##############
# Choice map #
##############


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

    @dispatch
    def merge(self, other: "BuiltinChoiceMap"):
        new_inner = self.trie.merge(other.trie)
        return BuiltinChoiceMap(new_inner)

    @dispatch
    def merge(self, other: EmptyChoiceMap):
        return self

    @dispatch
    def merge(self, other: ChoiceMap):
        raise Exception(
            f"Merging with choice map type {type(other)} not supported.",
        )

    def __setitem__(self, k, v):
        v = (
            ValueChoiceMap(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.trie.trie_insert(k, v)

    def __hash__(self):
        return hash(self.trie)


##############
# Shorthands #
##############

choice_map = BuiltinChoiceMap.new
select = BuiltinSelection.new
select_with = BuiltinSelection.with_selections
