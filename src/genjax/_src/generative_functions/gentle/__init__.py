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
"""The `genjax.gentle` language is a scratch / pedagogical generative function
language which exposes a less restrictive set of program constructs, based on
normal Python.

The downside of the `genjax.gentle` language is that you cannot invoke  its generative functions in JAX generative function code.
"""

import abc
import functools
import itertools
from dataclasses import dataclass

import jax.random as jr

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import NoneSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


_GENTLE_STACK = []


def handle(msg):
    assert _GENTLE_STACK
    handler = _GENTLE_STACK[-1]
    v = handler.process_message(msg)
    return v


class Handler(object):
    def __enter__(self):
        _GENTLE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            assert _GENTLE_STACK[-1] is self
            _GENTLE_STACK.pop()
        else:
            if self in _GENTLE_STACK:
                loc = _GENTLE_STACK.index(self)
                for _ in range(loc, len(_GENTLE_STACK)):
                    _GENTLE_STACK.pop()

    @abc.abstractmethod
    def process_message(self, msg):
        pass


def trace(addr: Any, gen_fn: GenerativeFunction, *args: Any) -> Any:
    # Must be handled.
    assert _GENTLE_STACK

    initial_msg = {
        "type": "trace",
        "addr": addr,
        "gen_fn": gen_fn,
        "args": args,
    }

    return handle(initial_msg)


# Usage: checks for duplicate addresses.
@dataclass
class AddressVisitor:
    visited: List

    @classmethod
    def new(cls):
        return AddressVisitor([])

    def visit(self, addr):
        if addr in self.visited:
            raise Exception(f"Already visited the address {addr}.")
        else:
            self.visited.append(addr)

    def merge(self, other):
        new = AddressVisitor.new()
        for addr in itertools.chain(self.visited, other.visited):
            new.visit(addr)


@dataclass
class SimulateHandler(Handler):
    key: PRNGKey
    choice_state: Trie
    trace_visitor: AddressVisitor
    score: FloatArray

    @classmethod
    def new(cls, key: PRNGKey):
        return SimulateHandler(
            key,
            Trie.new(),
            AddressVisitor.new(),
            0.0,
        )

    def process_message(self, msg):
        gen_fn = msg["gen_fn"]
        args = msg["args"]
        addr = msg["addr"]
        self.trace_visitor.visit(addr)
        self.key, tr = gen_fn.simulate(self.key, args)
        retval = tr.get_retval()
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        return retval


@dataclass
class GentleSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @typecheck
    @classmethod
    def new(cls, *addrs):
        trie = Trie.new()
        for addr in addrs:
            trie[addr] = AllSelection()
        return GentleSelection(trie)

    @typecheck
    @classmethod
    def with_selections(cls, selections: Dict):
        assert isinstance(selections, Dict)
        trie = Trie.new()
        for (k, v) in selections.items():
            assert isinstance(v, Selection)
            trie.trie_insert(k, v)
        return GentleSelection(trie)

    def filter(self, tree):
        def _inner(k, v):
            sub = self.trie[k]
            if sub is None:
                sub = NoneSelection()

            # Handles hierarchical in Trie.
            elif isinstance(sub, Trie):
                sub = GentleSelection(sub)

            under = sub.filter(v)
            return k, under

        trie = Trie.new()
        iter = tree.get_subtrees_shallow()
        for (k, v) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        return GentleChoiceMap(trie)

    def complement(self):
        return GentleComplementSelection(self.trie)

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
class GentleComplementSelection(Selection):
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
                sub = GentleSelection(sub)

            under = sub.complement().filter(v)
            return k, under

        trie = Trie.new()
        iter = chm.get_subtrees_shallow()
        for (k, v) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        return GentleChoiceMap(trie)

    def complement(self):
        return GentleSelection(self.trie)

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
class GentleChoiceMap(ChoiceMap):
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
        return GentleChoiceMap(trie)

    def is_empty(self):
        return self.trie.is_empty()

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
        return GentleSelection(trie)

    @dispatch
    def merge(self, other: "GentleChoiceMap"):
        new_inner, discard = self.trie.merge(other.trie)
        return GentleChoiceMap(new_inner), GentleChoiceMap(discard)

    @dispatch
    def merge(self, other: EmptyChoiceMap):
        return self, EmptyChoiceMap()

    @dispatch
    def merge(self, other: ChoiceMap):
        raise Exception(
            f"Merging with choice map type {type(other)} not supported.",
        )

    ###########
    # Dunders #
    ###########

    def __setitem__(self, k, v):
        v = (
            ValueChoiceMap(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.trie.trie_insert(k, v)

    def __hash__(self):
        return hash(self.trie)


@dataclass
class GentleTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Trie
    score: FloatArray

    def flatten(self):
        return (self.gen_fn, self.args, self.retval, self.choices, self.score), ()

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return GentleChoiceMap(self.choices)

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def project(self, selection: Selection):
        return 0.0


@dataclass
class GentleGenerativeFunction(GenerativeFunction):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> GentleTrace:
        with SimulateHandler.new(key) as handler:
            retval = self.source(*args)
            score = handler.score
            choices = handler.choice_state
            return key, GentleTrace(self, args, retval, choices, score)

    def importance(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, GentleTrace]:
        pass

    def update(
        self,
        key: PRNGKey,
        prev: GentleTrace,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Any, FloatArray, GentleTrace, ChoiceMap]:
        pass

    def assess(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Any, FloatArray]:
        pass


@typecheck
def gen(source: Callable):
    return GentleGenerativeFunction(source)
