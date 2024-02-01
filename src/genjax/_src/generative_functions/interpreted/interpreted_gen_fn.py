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


# Implement the Generative Function Interface using an effect handler style
# implementation (c.f. Pyro's [`poutine`](https://docs.pyro.ai/en/stable/poutine.html)
# for instance, although the code in this module is quite readable and localized).

import itertools
from dataclasses import dataclass

import jax
from beartype import beartype
from equinox import module_update_wrapper

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
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
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
_INTERPRETED_STACK: List["Handler"] = []


# A `Handler` implements Python's context manager protocol.
# It must also provide an implementation for `process_message`.
class Handler(object):
    def __enter__(self):
        _INTERPRETED_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            p = _INTERPRETED_STACK.pop()
            assert p is self
        else:
            try:
                loc = _INTERPRETED_STACK.index(self)
                del _INTERPRETED_STACK[loc:]
            except ValueError:
                pass

    def handle(self, gen_fn: GenerativeFunction, args: Tuple, addr: Any):
        raise NotImplementedError


# A primitive used in our language to denote invoking another generative function.
# Its behavior depends on the handler which is at the top of the stack
# when the primitive is invoked.
def trace(addr: Any, gen_fn: GenerativeFunction) -> Callable:
    """Invoke a generative function, binding its generative semantics with the current
    caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of [InterpretedGenerativeFunction].

    Returns:
        callable: A callable which wraps the `trace_p` primitive, accepting arguments
        (`args`) and binding the primitive with them. This raises the primitive to be
        handled by [`InterpretiveGenerativeFunction`][]
        transformations.
    """
    assert _INTERPRETED_STACK

    def invoke(*args: Tuple):
        assert _INTERPRETED_STACK
        handler = _INTERPRETED_STACK[-1]
        return handler.handle(gen_fn, args, addr)

    # Defer the behavior of this call to the handler.
    return invoke


# Usage: checks for duplicate addresses, which violates Gen's rules.
class AddressVisitor(Pytree):
    visited: List = Pytree.field(default_factory=list)

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


@dataclass
class SimulateHandler(Handler):
    key: PRNGKey
    score: FloatArray = 0.0
    choice_state: Trie = Pytree.field(default_factory=Trie)
    trace_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def handle(self, gen_fn: GenerativeFunction, args: Tuple, addr: Any):
        self.trace_visitor.visit(addr)
        self.key, sub_key = jax.random.split(self.key)
        tr = gen_fn.simulate(sub_key, args)
        retval = tr.get_retval()
        self.choice_state = self.choice_state.trie_insert(addr, tr)
        self.score += tr.get_score()
        return retval


@dataclass
class ImportanceHandler(Handler):
    key: PRNGKey
    constraints: ChoiceMap
    score: FloatArray = 0.0
    weight: FloatArray = 0.0
    choice_state: Trie = Pytree.field(default_factory=Trie)
    trace_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def handle(self, gen_fn: GenerativeFunction, args: Tuple, addr: Any):
        self.trace_visitor.visit(addr)
        sub_map = self.constraints.get_submap(addr)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w) = gen_fn.importance(sub_key, sub_map, args)
        retval = tr.get_retval()
        self.choice_state = self.choice_state.trie_insert(addr, tr)
        self.score += tr.get_score()
        self.weight += w
        return retval


@dataclass
class UpdateHandler(Handler):
    key: PRNGKey
    previous_trace: Trace
    constraints: ChoiceMap
    weight: FloatArray = 0.0
    discard: Trie = Pytree.field(default_factory=Trie)
    choice_state: Trie = Pytree.field(default_factory=Trie)
    trace_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def handle(self, gen_fn: GenerativeFunction, args: Tuple, addr: Any):
        self.trace_visitor.visit(addr)
        sub_map = self.constraints.get_submap(addr)
        # sub_trace = self.previous_trace.get_choice().get_submap(addr)
        # TODO(colin): think about this with McCoy. Having get_choice() implicitly
        # call strip() makes a _lot_ of interpreted code the same as the static code,
        # and it seems good not to ask people to add or remove strip() as they move
        # from one to another, and also means the type of thing you get from get_choice()
        # is more stable. Maybe we can move get_subtrace higher in the stack?
        if st := getattr(self.previous_trace, "get_subtrace", None):
            sub_trace = st(addr)
        else:
            sub_trace = self.previous_trace.get_choice().get_submap(addr)
        argdiffs = Diff.tree_diff_unknown_change(args)
        self.key, sub_key = jax.random.split(self.key)
        # if isinstance(sub_map, EmptyChoice):
        #     sub_map = HierarchicalChoiceMap.new({})
        (tr, w, rd, d) = gen_fn.update(sub_key, sub_trace, sub_map, argdiffs)
        retval = tr.get_retval()
        self.weight += w
        self.choice_state = self.choice_state.trie_insert(addr, tr)
        self.discard = self.discard.trie_insert(addr, d)
        return retval


@dataclass
class AssessHandler(Handler):
    constraints: ChoiceMap
    score: FloatArray = 0.0
    trace_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def handle(self, gen_fn: GenerativeFunction, args: Tuple, addr: Any):
        self.trace_visitor.visit(addr)
        sub_map = self.constraints.get_submap(addr)
        (score, retval) = gen_fn.assess(sub_map, args)
        self.score += score
        return retval


########################
# Generative datatypes #
########################


class InterpretedTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Trie
    score: FloatArray

    def get_gen_fn(self):
        return self.gen_fn

    def get_choice(self):
        return HierarchicalChoiceMap(self.choices).strip()

    def get_subtrace(self, addr):
        return self.choices[addr]

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def project(self, selection: HierarchicalSelection) -> FloatArray:
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
class InterpretedGenerativeFunction(GenerativeFunction, SupportsCalleeSugar):
    """An `InterpretedGenerativeFunction` is a generative function which relies only
    upon the CPU for its execution. This is in contrast to `static`,
    which is designed to enable [JAX acceleration](https://jax.readthedocs.io/en/latest/)
    for the inference computations.

    `InterpretedGenerativeFunction`s are easier to write: You can use natural
    Python flow control in your generative functions, and can work with arrays
    and structures of arbitrary shape, even having the shapes of matrices involved
    in your computations be random variables themselves. While such programs
    cannot take advantage of JAX, it may be a more comfortable environment for
    rapid prototyping or pedagogical work.

    Exploiting JAX requires more planning in the design of the generative functions,
    since the sizes of arrays, etc., must be known in advance to take advantage
    of GPU-style acceleration, and forks in the road in the implementation of
    such functions must be represented in the linear algebra of the code.

    Furthermore, you must prepare your execution environment with a version of
    [jaxlib](https://jax.readthedocs.io/en/latest/installation.html) which
    can expose the acceleration features of your hardware environment to JAX.

    In the meantime, you can work in the interpreted Gen dialect and postpone
    the effort of integrating with JAX, working with the Gen paradigm in an
    non-accelerated form.

    To create an [`InterpretedGenerativeFunction`][], use the [`interpreted`][]
    decorator like this:

        ```python
        import genjax


        @genjax.interpreted
        def model():
            y = genjax.normal(0.0, 1.0) @ "y"
            return y
        ```
    """

    source: Callable = Pytree.static()

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
    ) -> Tuple[InterpretedTrace, FloatArray]:
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
    ) -> Tuple[InterpretedTrace, FloatArray, Any, ChoiceMap]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        if isinstance(choice_map, EmptyChoice):
            choice_map = HierarchicalChoiceMap()
        with UpdateHandler(key, prev_trace, choice_map) as handler:
            args = Diff.tree_primal(argdiffs)
            retval = syntax_sugar_handled(*args)
            choices = handler.choice_state
            weight = handler.weight
            discard = handler.discard
            retdiff = Diff.tree_diff_unknown_change(retval)
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
    ) -> Tuple[FloatArray, Any]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        with AssessHandler(choice_map) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            return score, retval

    def inline(self, *args):
        return self.source(*args)

    @property
    def __wrapped__(self):
        return self.source


#############
# Decorator #
#############


def interpreted_gen_fn(f) -> InterpretedGenerativeFunction:
    return module_update_wrapper(InterpretedGenerativeFunction(f))
