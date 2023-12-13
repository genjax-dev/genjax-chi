# Copyright 2023 MIT Probabilistic Computing Project
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
"""The `genjax.Interpreted` language is a generative function language which
exposes a less restrictive set of program constructs, based on normal Python programs. It implements the GFI using an effect handler style implementation (c.f. Pyro's [`poutine`](https://docs.pyro.ai/en/stable/poutine.html) for instance, although the code in this module is quite readable and localized).

The intent of this language is pedagogical - one can use it to rapidly construct models and prototype inference, but it is not intended to be used for performance critical applications, for several reasons:

* Instances of `genjax.Interpreted` generative functions *cannot* be invoked as callees within JAX generative function code, which prevents compositional usage (from above, within `JAXGenerativeFunction` instances).

* It does not feature gradient interfaces - supporting an ad hoc Python AD implementation is out of scope for the intended applications of GenJAX.
"""

import abc
import itertools
from dataclasses import dataclass

import jax

from genjax._src.core.datatypes.generative import ChoiceMap, Choice
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalChoiceMap
from genjax._src.core.datatypes.generative import LanguageConstructor
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.incremental import (
    UnknownChange,
    tree_diff_unknown_change,
)
from genjax._src.core.interpreters.incremental import tree_diff
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.typing import Any, FloatArray
from genjax._src.core.typing import Callable
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.supports_callees import (
    push_trace_overload_stack,
    SupportsCalleeSugar,
)

# Our main idiom to express non-standard interpretation is an
# (effect handler)-inspired dispatch stack.
_INTERPRETED_STACK = []


# When `handle` is invoked, it dispatches the information in `msg`
# to the handler at the top of the stack (end of list).
def handle(msg):
    assert _INTERPRETED_STACK
    handler = _INTERPRETED_STACK[-1]
    v = handler.process_message(msg)
    return v


# A `Handler` implements Python's context manager protocol.
# It must also provide an implementation for `process_message`.
class Handler(object):
    def __enter__(self):
        _INTERPRETED_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            assert _INTERPRETED_STACK[-1] is self
            _INTERPRETED_STACK.pop()
        else:
            if self in _INTERPRETED_STACK:
                loc = _INTERPRETED_STACK.index(self)
                for _ in range(loc, len(_INTERPRETED_STACK)):
                    _INTERPRETED_STACK.pop()

    @abc.abstractmethod
    def process_message(self, msg):
        pass


# A primitive used in our language to denote invoking another generative function.
# It's behavior depends on the handler which is at the top of the stack
# when the primitive is invoked.
def trace(addr: Any, gen_fn: GenerativeFunction) -> Callable:
    # Must be handled.
    assert _INTERPRETED_STACK

    def invoke(*args: Tuple):
        return handle(
            {
                "type": "trace",
                "addr": addr,
                "gen_fn": gen_fn,
                "args": args,
            }
        )

    # Defer the behavior of this call to the handler.
    return invoke


# Usage: checks for duplicate addresses, which violates Gen's rules.
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


#####################################
# Generative semantics via handlers #
#####################################


@dataclass
class SimulateHandler(Handler):
    key: PRNGKey
    score: float
    choice_state: Trie
    trace_visitor: AddressVisitor

    @classmethod
    def new(cls, key: PRNGKey):
        return SimulateHandler(
            key,
            0.0,
            Trie.new(),
            AddressVisitor.new(),
        )

    def process_message(self, msg):
        gen_fn = msg["gen_fn"]
        args = msg["args"]
        addr = msg["addr"]
        self.trace_visitor.visit(addr)
        self.key, sub_key = jax.random.split(self.key)
        tr = gen_fn.simulate(sub_key, args)
        retval = tr.get_retval()
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        return retval


@dataclass
class ImportanceHandler(Handler):
    key: PRNGKey
    score: float
    weight: float
    constraints: ChoiceMap
    choice_state: Trie
    trace_visitor: AddressVisitor

    @classmethod
    def new(cls, key: PRNGKey, constraints: ChoiceMap):
        return ImportanceHandler(
            key,
            0.0,
            0.0,
            constraints,
            Trie.new(),
            AddressVisitor.new(),
        )

    def process_message(self, msg):
        gen_fn = msg["gen_fn"]
        args = msg["args"]
        addr = msg["addr"]
        self.trace_visitor.visit(addr)
        sub_map = self.constraints.get_submap(addr)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w) = gen_fn.importance(sub_key, sub_map, args)
        retval = tr.get_retval()
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        self.weight += w
        return retval


@dataclass
class UpdateHandler(Handler):
    key: PRNGKey
    weight: float
    previous_trace: Trace
    constraints: ChoiceMap
    discard: Trie
    choice_state: Trie
    trace_visitor: AddressVisitor

    @classmethod
    def new(cls, key: PRNGKey, previous_trace: Trace, constraints: ChoiceMap):
        return UpdateHandler(
            key,
            0.0,
            previous_trace,
            constraints,
            Trie.new(),
            Trie.new(),
            AddressVisitor.new(),
        )

    def process_message(self, msg):
        gen_fn = msg["gen_fn"]
        args = msg["args"]
        addr = msg["addr"]
        self.trace_visitor.visit(addr)
        sub_map = self.constraints.get_submap(addr)
        sub_trace = self.previous_trace.get_choices().get_submap(addr)
        argdiffs = tree_diff_unknown_change(args)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w, rd, d) = gen_fn.update(sub_key, sub_trace, sub_map, argdiffs)
        retval = tr.get_retval()
        self.weight += w
        self.choice_state[addr] = tr
        self.discard[addr] = d
        return retval


@dataclass
class AssessHandler(Handler):
    score: FloatArray | float
    constraints: ChoiceMap
    trace_visitor: AddressVisitor

    @classmethod
    def new(cls, constraints: ChoiceMap):
        return AssessHandler(
            0.0,
            constraints,
            AddressVisitor.new(),
        )

    def process_message(self, msg):
        gen_fn = msg["gen_fn"]
        args = msg["args"]
        addr = msg["addr"]
        self.trace_visitor.visit(addr)
        sub_map = self.constraints.get_submap(addr)
        (score, retval) = gen_fn.assess(sub_map, args)
        self.score += score
        return retval


########################
# Generative datatypes #
########################


@dataclass
class InterpretedTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Trie
    score: float | FloatArray

    def flatten(self):
        return (self.gen_fn, self.args, self.retval, self.choices, self.score), ()

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return HierarchicalChoiceMap(self.choices).strip()

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def project(self, selection: Selection):
        return 0.0


# Callee syntactic sugar handler.
@typecheck
def handler_trace_with_interpreted(addr, gen_fn: GenerativeFunction, args: Tuple):
    return trace(addr, gen_fn)(*args)


# Our generative function type - simply wraps a `source: Callable`
# which can invoke our `trace` primitive.
@dataclass
class InterpretedGenerativeFunction(GenerativeFunction, SupportsCalleeSugar):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    @classmethod
    @typecheck
    def new(cls, callable: Callable):
        return InterpretedGenerativeFunction(callable)

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> InterpretedTrace:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        # Handle trace with the `SimulateHandler`.
        with SimulateHandler.new(key) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            choices = handler.choice_state
            return InterpretedTrace(self, args, retval, choices, score)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[InterpretedTrace, FloatArray | float]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        with ImportanceHandler.new(key, choice_map) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            choices = handler.choice_state
            weight = handler.weight
            return (
                InterpretedTrace(self, args, retval, choices, score),
                weight,
            )

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev_trace: InterpretedTrace,
        choice_map: Choice,
        argdiffs: Tuple,
    ) -> Tuple[InterpretedTrace, float, Any, ChoiceMap]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        with UpdateHandler.new(key, prev_trace, choice_map) as handler:
            args = tree_diff_primal(argdiffs)
            retval = syntax_sugar_handled(*args)
            choices = handler.choice_state
            weight = handler.weight
            discard = handler.discard
            retdiff = tree_diff(retval, UnknownChange)
            score = prev_trace.get_score() + weight
            return (
                InterpretedTrace(self, args, retval, choices, score),
                weight,
                retdiff,
                HierarchicalChoiceMap(discard),
            )

    @typecheck
    def assess(
        self,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray | float, Any]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        with AssessHandler.new(choice_map) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            return score, retval

    def inline(self, *args):
        return self.source(*args)


########################
# Language constructor #
########################


def interpreted_gen_fn(source: Callable):
    return InterpretedGenerativeFunction.new(source)


Interpreted = LanguageConstructor(
    interpreted_gen_fn,
)
