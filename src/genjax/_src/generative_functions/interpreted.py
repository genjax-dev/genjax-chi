# Copyright 2024 MIT Probabilistic Computing Project
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

from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Address,
    Argdiffs,
    ChoiceMap,
    GenerativeFunction,
    Retdiff,
    Sample,
    Score,
    Selection,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.generative.core import push_trace_overload_stack
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    List,
    PRNGKey,
    Tuple,
    typecheck,
)


class AddressReuse(Exception):
    """Attempt to re-write an address in a GenJAX trace.

    Any given address for a random choice may only be written to once. You can choose a
    different name for the choice, or nest it into a scope where it is unique.
    """


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

    @abstractmethod
    def handle(self, addr: Address, gen_fn: GenerativeFunction, args: Tuple):
        raise NotImplementedError


# A primitive used in our language to denote invoking another generative function.
# Its behavior depends on the handler which is at the top of the stack
# when the primitive is invoked.
def trace(addr: Any, gen_fn: GenerativeFunction) -> Callable[..., Any]:
    """Invoke a generative function, binding its generative semantics with the current
    caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee.

    Returns:
        callable: A callable which wraps the `trace_p` primitive, accepting arguments
        (`args`) and binding the primitive with them. This raises the primitive to be
        handled by transformations.
    """
    assert _INTERPRETED_STACK

    def invoke(*args: Tuple):
        assert _INTERPRETED_STACK
        handler = _INTERPRETED_STACK[-1]
        return handler.handle(addr, gen_fn, args)

    # Defer the behavior of this call to the handler.
    return invoke


# Usage: checks for duplicate addresses, which violates Gen's rules.
@Pytree.dataclass
class AddressVisitor(Pytree):
    visited: List = Pytree.static(default_factory=list)

    def visit(self, addr):
        if addr in self.visited:
            raise AddressReuse(addr)
        else:
            self.visited.append(addr)

    def get_visited(self):
        return self.visited


#####################################
# Generative semantics via handlers #
#####################################


@dataclass
class SimulateHandler(Handler):
    key: PRNGKey
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    address_traces: List[Trace] = Pytree.field(default_factory=list)

    def visit(self, addr):
        self.address_visitor.visit(addr)

    @typecheck
    def handle(self, addr: Address, gen_fn: GenerativeFunction, args: Tuple):
        self.visit(addr)
        self.key, sub_key = jax.random.split(self.key)
        tr = gen_fn.simulate(sub_key, args)
        self.address_traces.append(tr)
        self.score += tr.get_score()
        v = tr.get_retval()
        return v


@dataclass
class UpdateHandler(Handler):
    key: PRNGKey
    previous_trace: Trace
    fwd_problem: UpdateProblem
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_problems: List[UpdateProblem] = Pytree.field(default_factory=list)

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def get_subproblem(self, addr: Address):
        match self.fwd_problem:
            case ChoiceMap():
                return self.fwd_problem.get_submap(addr)

            case Selection():
                subproblem = self.fwd_problem.step(addr)
                return subproblem

            case _:
                raise ValueError(f"Not implemented fwd_problem: {self.fwd_problem}")

    def get_subtrace(self, addr: Address):
        return self.previous_trace.get_subtrace(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Diff))

    @typecheck
    def handle(
        self,
        addr: Address,
        gen_fn: GenerativeFunction,
        argdiffs: Tuple,
    ):
        self.visit(addr)
        subtrace = self.get_subtrace(addr)
        subproblem = self.get_subproblem(addr)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w, retval_diff, bwd_problem) = gen_fn.update(
            sub_key, subtrace, subproblem, argdiffs
        )
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_problems.append(bwd_problem)

        return retval_diff


@dataclass
class AssessHandler(Handler):
    sample: Sample
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def get_subsample(self, addr: Address):
        match self.sample:
            case ChoiceMap():
                return self.sample.get_submap(addr)

            case _:
                raise ValueError(f"Not implemented: {self.sample}")

    @typecheck
    def handle(self, addr: Address, gen_fn: GenerativeFunction, args: Tuple):
        submap = self.get_subsample(addr)
        (score, v) = gen_fn.assess(submap, args)
        self.score += score
        return v


########################
# Generative datatypes #
########################


@Pytree.dataclass
class InterpretedTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    addresses: AddressVisitor
    subtraces: List[Trace]
    score: FloatArray

    def get_gen_fn(self):
        return self.gen_fn

    def get_sample(self) -> ChoiceMap:
        addresses = self.addresses.get_visited()
        chm = ChoiceMap.n
        for addr, subtrace in zip(addresses, self.subtraces):
            chm = chm ^ ChoiceMap.a(addr, subtrace.get_sample())

        return chm

    def get_subtrace(self, addr):
        return self.choices[addr]

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        weight = jnp.zeros(())
        for k, subtrace in self.choices.get_submaps_shallow():
            key, sub_key = jax.random.split(key)
            weight += subtrace.project(sub_key, selection.step(k))
        return weight


# Callee syntactic sugar handler.
@typecheck
def handler_trace_with_interpreted(
    addr: Address,
    gen_fn: GenerativeFunction,
    args: Tuple,
):
    return trace(addr, gen_fn)(args)


# Our generative function type - simply wraps a `source: Callable[..., Any]`
# which can invoke our `trace` primitive.
@Pytree.dataclass
class InterpretedGenerativeFunction(GenerativeFunction):
    """An `InterpretedGenerativeFunction` is a generative function which supports a permissive subset of Python for its modeling language. Permissive here is in contrast to the `StaticGenerativeFunction` language, which supports similar modeling abstractions, but requires that users write within the JAX compatible subset of Python -
    designed to enable [JAX acceleration](https://jax.readthedocs.io/en/latest/)
    for the inference computations.

    `InterpretedGenerativeFunction`s are easy to write: you can use natural
    Python flow control in your generative functions, and can work with arrays
    and structures of arbitrary shape, even having the shapes of matrices involved
    in your computations be random variables themselves. While such programs
    cannot take advantage of JAX, it may be a comfortable environment for
    rapid prototyping or pedagogical work.

    Exploiting JAX requires more planning in the design of the generative functions,
    since the sizes of arrays, etc., must be known in advance to take advantage
    of GPU-style acceleration.

    Furthermore, you must prepare your execution environment with a version of
    [jaxlib](https://jax.readthedocs.io/en/latest/installation.html) which
    can expose the acceleration features of your hardware environment to JAX.

    In the meantime, you can work in the interpreted Gen dialect and postpone
    the effort of integrating with JAX, working with the Gen paradigm in an
    non-accelerated form.

    To create an `InterpretedGenerativeFunction`, you can use the `interpreted_gen_fn`
    decorator like this:

    ```python
    import genjax


    @genjax.interpreted_gen_fn
    def model():
        y = genjax.normal(0.0, 1.0) @ "y"
        return y
    ```
    """

    source: Callable[..., Any] = Pytree.static()

    @GenerativeFunction.gfi_boundary
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
        with SimulateHandler(key) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            visitor = handler.address_visitor
            traces = handler.address_traces
            return InterpretedTrace(self, args, retval, visitor, traces, score)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[InterpretedTrace, Weight, Retdiff, ChoiceMap]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )

        def make_bwd_problem(visitor, subproblems):
            addresses = visitor.get_visited()
            addresses = Pytree.tree_unwrap_const(addresses)
            chm = ChoiceMap.n
            for addr, subproblem in zip(addresses, subproblems):
                chm = chm ^ ChoiceMap.a(addr, subproblem)
            return chm

        with UpdateHandler(key, trace, update_problem) as handler:
            args = Diff.tree_primal(argdiffs)
            retval = syntax_sugar_handled(*args)
            visitor = handler.address_visitor
            traces = handler.address_traces
            weight = handler.weight
            score = handler.score
            retdiff = Diff.tree_diff_unknown_change(retval)
            bwd_problem = make_bwd_problem(visitor, handler.bwd_problems)
            return (
                InterpretedTrace(self, args, retval, visitor, traces, score),
                weight,
                retdiff,
                bwd_problem,
            )

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_interpreted, self.source
        )
        with AssessHandler(sample) as handler:
            retval = syntax_sugar_handled(*args)
            score = handler.score
            return score, retval

    def inline(self, *args):
        return self.source(*args)


#############
# Decorator #
#############


def gen(f) -> InterpretedGenerativeFunction:
    return InterpretedGenerativeFunction(f)
