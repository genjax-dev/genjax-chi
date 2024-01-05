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
import functools
import itertools
from dataclasses import dataclass, field

import jax
import jaxtyping
from beartype import beartype

from genjax._src.core.datatypes.generative import (
    Choice,
    ChoiceMap,
    EmptyChoice,
    GenerativeFunction,
    HierarchicalChoiceMap,
    HierarchicalSelection,
    Trace,
)
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.incremental import (
    UnknownChange,
    tree_diff,
    tree_diff_primal,
    tree_diff_unknown_change,
)
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    FloatArray,
    List,
    PRNGKey,
    Tuple,
)
from genjax._src.generative_functions.supports_callees import (
    SupportsCalleeSugar,
    push_trace_overload_stack,
)
from genjax.core.exceptions import AddressReuse

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
@dataclass(eq=False)
@beartype
class AddressVisitor:
    visited: List = field(default_factory=list)

    def visit(self, addr):
        if addr in self.visited:
            raise AddressReuse(addr)
        else:
            self.visited.append(addr)

    def merge(self, other):
        new = AddressVisitor()
        for addr in itertools.chain(self.visited, other.visited):
            new.visit(addr)


#####################################
# Generative semantics via handlers #
#####################################


@dataclass(eq=False)
@beartype
class SimulateHandler(Handler):
    key: PRNGKey
    score: ArrayLike = 0.0
    choice_state: Trie = field(default_factory=Trie)
    trace_visitor: AddressVisitor = field(default_factory=AddressVisitor)

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


@dataclass(eq=False)
@beartype
class ImportanceHandler(Handler):
    key: PRNGKey
    constraints: ChoiceMap
    score: ArrayLike = 0.0
    weight: ArrayLike = 0.0
    choice_state: Trie = field(default_factory=Trie)
    trace_visitor: AddressVisitor = field(default_factory=AddressVisitor)

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


@dataclass(eq=False)
@beartype
class UpdateHandler(Handler):
    key: PRNGKey
    previous_trace: Trace
    constraints: ChoiceMap
    weight: ArrayLike = 0.0
    discard: Trie = field(default_factory=Trie)
    choice_state: Trie = field(default_factory=Trie)
    trace_visitor: AddressVisitor = field(default_factory=AddressVisitor)

    def process_message(self, msg):
        gen_fn = msg["gen_fn"]
        args = msg["args"]
        addr = msg["addr"]
        self.trace_visitor.visit(addr)
        sub_map = self.constraints.get_submap(addr)
        # sub_trace = self.previous_trace.get_choices().get_submap(addr)
        # TODO(colin): think about this with McCoy. Having get_choices() implicitly
        # call strip() makes a _lot_ of interpreted code the same as the static code,
        # and it seems good not to ask people to add or remove strip() as they move
        # from one to another, and also means the type of thing you get from get_choices()
        # is more stable. Maybe we can move get_subtrace higher in the stack?
        if st := getattr(self.previous_trace, "get_subtrace", None):
            sub_trace = st(addr)
        else:
            sub_trace = self.previous_trace.get_choices().get_submap(addr)
        argdiffs = tree_diff_unknown_change(args)
        self.key, sub_key = jax.random.split(self.key)
        # if isinstance(sub_map, EmptyChoice):
        #     sub_map = HierarchicalChoiceMap.new({})
        (tr, w, rd, d) = gen_fn.update(sub_key, sub_trace, sub_map, argdiffs)
        retval = tr.get_retval()
        self.weight += w
        self.choice_state[addr] = tr
        self.discard[addr] = d
        return retval


@dataclass(eq=False)
@beartype
class AssessHandler(Handler):
    constraints: ChoiceMap
    score: ArrayLike = 0.0
    trace_visitor: AddressVisitor = field(default_factory=AddressVisitor)

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
@beartype
class InterpretedTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Trie
    score: jaxtyping.ArrayLike

    def flatten(self):
        return (self.gen_fn, self.args, self.retval, self.choices, self.score), ()

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return HierarchicalChoiceMap(self.choices).strip()

    def get_subtrace(self, addr):
        return self.choices[addr]

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def project(self, selection: HierarchicalSelection) -> ArrayLike:
        weight = 0.0
        for k, subtrace in self.choices.get_submaps_shallow():
            if selection.has_addr(k):
                weight += subtrace.project(selection.get_subselection(k))
        return weight


# Callee syntactic sugar handler.
@beartype
def handler_trace_with_interpreted(addr, gen_fn: GenerativeFunction, args: Tuple):
    return trace(addr, gen_fn)(*args)


# Our generative function type - simply wraps a `source: Callable`
# which can invoke our `trace` primitive.
@dataclass
@beartype
class InterpretedGenerativeFunction(GenerativeFunction, SupportsCalleeSugar):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> InterpretedTrace:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        # Handle trace with the `SimulateHandler`.
        with SimulateHandler(key) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            choices = handler.choice_state
            return InterpretedTrace(self, args, retval, choices, score)

    def importance(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[InterpretedTrace, jaxtyping.ArrayLike]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        with ImportanceHandler(key, choice_map) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            choices = handler.choice_state
            weight = handler.weight
            return (
                InterpretedTrace(self, args, retval, choices, score),
                weight,
            )

    def update(
        self,
        key: PRNGKey,
        prev_trace: Trace,
        choice_map: Choice,
        argdiffs: Tuple,
    ) -> Tuple[InterpretedTrace, ArrayLike, Any, ChoiceMap]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        if isinstance(choice_map, EmptyChoice):
            choice_map = HierarchicalChoiceMap()
        with UpdateHandler(key, prev_trace, choice_map) as handler:
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

    def assess(
        self,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray | float, Any]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        with AssessHandler(choice_map) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            return score, retval

    def inline(self, *args):
        return self.source(*args)


#############
# Decorator #
#############


def Interpreted(f) -> InterpretedGenerativeFunction:
    gf = InterpretedGenerativeFunction(f)
    functools.update_wrapper(gf, f)
    return gf
