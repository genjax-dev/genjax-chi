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

import functools
from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    ChoiceMapBuilder,
    EditRequest,
    EmptyRequest,
    EmptyTrace,
    GenerativeFunction,
    ImportanceRequest,
    IncrementalGenericRequest,
    Retdiff,
    Sample,
    Score,
    Selection,
    StaticAddress,
    StaticAddressComponent,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import R, push_trace_overload_stack
from genjax._src.core.interpreters.forward import (
    InitialStylePrimitive,
    StatefulHandler,
    forward,
    initial_style_bind,
)
from genjax._src.core.interpreters.incremental import (
    Diff,
    incremental,
)
from genjax._src.core.pytree import Closure, Const, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    PRNGKey,
)

_WRAPPER_ASSIGNMENTS = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
)


# Usage in transforms: checks for duplicate addresses.
@Pytree.dataclass
class AddressVisitor(Pytree):
    visited: list[StaticAddress] = Pytree.static(default_factory=list)

    def visit(self, addr: StaticAddress):
        if addr in self.visited:
            raise AddressReuse(addr)
        else:
            self.visited.append(addr)

    def get_visited(self):
        return self.visited


#########
# Trace #
#########


@Pytree.dataclass
class StaticTrace(Generic[R], Trace[R]):
    gen_fn: GenerativeFunction[R]
    args: tuple[Any, ...]
    retval: R
    addresses: AddressVisitor
    subtraces: list[Trace[Any]]
    score: Score

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> R:
        return self.retval

    def get_gen_fn(self) -> GenerativeFunction[R]:
        return self.gen_fn

    def get_sample(self) -> ChoiceMap:
        addresses = self.addresses.get_visited()
        chm = ChoiceMap.empty()
        for addr, subtrace in zip(addresses, self.subtraces):
            chm = chm ^ ChoiceMapBuilder.a(addr, subtrace.get_sample())

        return chm

    def get_score(self) -> Score:
        return self.score

    def get_subtrace(self, addr: StaticAddress):
        addresses = self.addresses.get_visited()
        idx = addresses.index(addr)
        return self.subtraces[idx]


##############################
# Static language exceptions #
##############################


class AddressReuse(Exception):
    """Attempt to re-write an address in a GenJAX trace.

    Any given address for a random choice may only be written to once. You can choose a
    different name for the choice, or nest it into a scope where it is unique.
    """


##############
# Primitives #
##############

# Generative function trace intrinsic.
trace_p = InitialStylePrimitive("trace")


############################################################
# Trace call (denotes invocation of a generative function) #
############################################################


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(
    _: tuple[Const[StaticAddress], ...],
    gen_fn: GenerativeFunction[R],
    args: tuple[Any, ...],
):
    return gen_fn.__abstract_call__(*args)


def trace(
    addr: StaticAddress,
    gen_fn: GenerativeFunction[R],
    args: tuple[Any, ...],
):
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of `StaticGenerativeFunction`.

    """
    addr = Pytree.tree_const(addr)
    return initial_style_bind(trace_p)(_abstract_gen_fn_call)(
        addr,
        gen_fn,
        args,
    )


######################################
#  Generative function interpreters  #
######################################


###########################
# Static language handler #
###########################


# This explicitly makes assumptions about some common fields:
# e.g. it assumes if you are using `StaticHandler.get_submap`
# in your code, that your derived instance has a `constraints` field.
@dataclass
class StaticHandler(StatefulHandler):
    @abstractmethod
    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction[R],
        args: tuple[Any, ...],
    ):
        raise NotImplementedError

    def handle_retval(self, v):
        return jtu.tree_leaves(v)

    # By default, the interpreter handlers for this language
    # handle the two primitives we defined above
    # (`trace_p`, for random choices)
    def handles(self, primitive):
        return primitive == trace_p

    def dispatch(self, primitive, *tracers, **_params):
        in_tree = _params["in_tree"]
        num_consts = _params.get("num_consts", 0)
        non_const_tracers = tracers[num_consts:]
        addr, gen_fn, args = jtu.tree_unflatten(in_tree, non_const_tracers)
        addr = Pytree.tree_const_unwrap(addr)
        if primitive == trace_p:
            v = self.handle_trace(addr, gen_fn, args)
            return self.handle_retval(v)
        else:
            raise Exception("Illegal primitive: {}".format(primitive))


############
# Simulate #
############


@dataclass
class SimulateHandler(StaticHandler):
    key: PRNGKey
    score: Score = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    address_traces: list[Trace[Any]] = Pytree.field(default_factory=list)

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def yield_state(self):
        return (
            self.address_visitor,
            self.address_traces,
            self.score,
        )

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        self.visit(addr)
        self.key, sub_key = jax.random.split(self.key)
        tr = gen_fn.simulate(sub_key, args)
        score = tr.get_score()
        self.address_traces.append(tr)
        self.score += score
        v = tr.get_retval()
        return v


def simulate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, args):
        stateful_handler = SimulateHandler(key)
        retval = forward(source_fn)(stateful_handler, *args)
        (
            address_visitor,
            address_traces,
            score,
        ) = stateful_handler.yield_state()
        return (
            args,
            retval,
            address_visitor,
            address_traces,
            score,
        )

    return wrapper


##########
# Update #
##########


@dataclass
class UpdateHandler(StaticHandler):
    key: PRNGKey
    previous_trace: EmptyTrace[Any] | StaticTrace[Any]
    fwd_problem: EditRequest
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: list[Trace[Any]] = Pytree.field(default_factory=list)
    bwd_problems: list[EditRequest] = Pytree.field(default_factory=list)

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_problems,
        )

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def get_subproblem(self, addr: StaticAddress):
        match self.fwd_problem:
            case ChoiceMap():
                return self.fwd_problem(addr)

            case ImportanceRequest(constraint) if isinstance(constraint, ChoiceMap):
                return ImportanceRequest(constraint(addr))

            case Selection():
                subproblem = self.fwd_problem(addr)
                return subproblem

            case EmptyRequest():
                return EmptyRequest()

            case _:
                raise ValueError(f"Not implemented fwd_problem: {self.fwd_problem}")

    def get_subtrace(self, sub_gen_fn: GenerativeFunction[Any], addr: StaticAddress):
        if isinstance(self.previous_trace, EmptyTrace):
            return EmptyTrace(sub_gen_fn)
        else:
            return self.previous_trace.get_subtrace(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Diff))

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        argdiffs: Argdiffs = args
        self.visit(addr)
        subtrace = self.get_subtrace(gen_fn, addr)
        subproblem = self.get_subproblem(addr)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w, retval_diff, bwd_problem) = gen_fn.edit(
            sub_key, subtrace, IncrementalGenericRequest(argdiffs, subproblem)
        )
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_problems.append(bwd_problem)

        return retval_diff


def update_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, previous_trace, constraints, diffs: tuple[Any, ...]):
        stateful_handler = UpdateHandler(key, previous_trace, constraints)
        diff_primals = Diff.tree_primal(diffs)
        diff_tangents = Diff.tree_tangent(diffs)
        retval_diffs = incremental(source_fn)(
            stateful_handler, diff_primals, diff_tangents
        )
        retval_primals = Diff.tree_primal(retval_diffs)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_problems,
        ) = stateful_handler.yield_state()
        return (
            (
                retval_diffs,
                weight,
                # Trace.
                (
                    diff_primals,
                    retval_primals,
                    address_visitor,
                    address_traces,
                    score,
                ),
                # Backward update problem.
                bwd_problems,
            ),
        )

    return wrapper


##########
# Assess #
##########


@dataclass
class AssessHandler(StaticHandler):
    sample: Sample
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def yield_state(self):
        return (self.score,)

    def get_subsample(self, addr: StaticAddress):
        match self.sample:
            case ChoiceMap():
                return self.sample(addr)

            case _:
                raise ValueError(f"Not implemented: {self.sample}")

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        submap = self.get_subsample(addr)
        (score, v) = gen_fn.assess(submap, args)
        self.score += score
        return v


def assess_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(constraints, args):
        stateful_handler = AssessHandler(constraints)
        retval = forward(source_fn)(stateful_handler, *args)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper


#######################
# Generative function #
#######################


# Callee syntactic sugar handler.


def handler_trace_with_static(
    addr: StaticAddressComponent | StaticAddress,
    gen_fn: GenerativeFunction[Any],
    args: tuple[Any, ...],
):
    return trace(addr if isinstance(addr, tuple) else (addr,), gen_fn, args)


@Pytree.dataclass
class StaticGenerativeFunction(Generic[R], GenerativeFunction[R]):
    """A `StaticGenerativeFunction` is a generative function which relies on program
    transformations applied to JAX-compatible Python programs to implement the generative
    function interface.

    By virtue of the implementation, any source program which is provided to this generative function *must* be JAX traceable, meaning [all the footguns for programs that JAX exposes](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) apply to the source program.

    **Language restrictions**

    In addition to JAX footguns, there are a few more which are specific to the generative function interface semantics. Here is the full list of language restrictions (and capabilities):

    * One is allowed to use `jax.lax` control flow primitives _so long as the functions provided to the primitives do not contain `trace` invocations_. In other words, utilizing control flow primitives within the source of a `StaticGenerativeFunction`'s source program requires that the control flow primitives get *deterministic* computation.

    * The above restriction also applies to `jax.vmap`.

    * Source programs are allowed to utilize untraced randomness, although there are restrictions (which we discuss below). It is required to use [`jax.random`](https://jax.readthedocs.io/en/latest/jax.random.html) and JAX's PRNG capabilities. To utilize untraced randomness, you'll need to pass in an extra key as an argument to your model.

        ```python
        @gen
        def model(key: PRNGKey):
            v = some_untraced_call(key)
            x = trace("x", genjax.normal)(v, 1.0)
            return x
        ```
    """

    source: Closure[R]
    """
    The source program of the generative function. This is a JAX-compatible Python program.
    """

    # To get the type of return value, just invoke
    # the source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        return self.source(*args)

    def __post_init__(self):
        wrapped = self.source.fn
        # Preserve the original function's docstring and name
        for k in _WRAPPER_ASSIGNMENTS:
            v = getattr(wrapped, k, None)
            if v is not None:
                object.__setattr__(self, k, v)

        object.__setattr__(self, "__wrapped__", wrapped)

    def handle_kwargs(self) -> "StaticGenerativeFunction[R]":
        @Pytree.partial()
        def kwarged_source(args, kwargs):
            return self.source(*args, **kwargs)

        return StaticGenerativeFunction(kwarged_source)

    @GenerativeFunction.gfi_boundary
    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> StaticTrace[R]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (args, retval, address_visitor, address_traces, score) = simulate_transform(
            syntax_sugar_handled
        )(key, args)
        return StaticTrace(
            self,
            args,
            retval,
            address_visitor,
            address_traces,
            score,
        )

    def edit_change_target(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[StaticTrace[R], Weight, Retdiff[R], EditRequest]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (
            (
                retval_diffs,
                weight,
                (
                    arg_primals,
                    retval_primals,
                    address_visitor,
                    address_traces,
                    score,
                ),
                bwd_problems,
            ),
        ) = update_transform(syntax_sugar_handled)(key, trace, edit_request, argdiffs)

        def make_bwd_problem(visitor, subproblems):
            addresses = visitor.get_visited()
            addresses = Pytree.tree_const_unwrap(addresses)
            chm = ChoiceMap.empty()
            for addr, subproblem in zip(addresses, subproblems):
                chm = chm ^ ChoiceMapBuilder.a(addr, subproblem)
            return chm

        bwd_problem = make_bwd_problem(address_visitor, bwd_problems)
        return (
            StaticTrace(
                self,
                arg_primals,
                retval_primals,
                address_visitor,
                address_traces,
                score,
            ),
            weight,
            retval_diffs,
            bwd_problem,
        )

    @GenerativeFunction.gfi_boundary
    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
    ) -> tuple[StaticTrace[R], Weight, Retdiff[R], EditRequest]:
        match edit_request:
            case IncrementalGenericRequest(argdiffs, subproblem):
                return self.edit_change_target(key, trace, subproblem, argdiffs)
            case _:
                return self.update(
                    key,
                    trace,
                    IncrementalGenericRequest(
                        Diff.no_change(trace.get_args()), edit_request
                    ),
                )

    @GenerativeFunction.gfi_boundary
    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (retval, score) = assess_transform(syntax_sugar_handled)(sample, args)
        return (score, retval)

    def inline(self, *args):
        return self.source(*args)

    def partial_apply(self, *args) -> "StaticGenerativeFunction[R]":
        """
        Returns a new [`StaticGenerativeFunction`][] with the given arguments partially applied.

        This method creates a new [`StaticGenerativeFunction`][] that has some of its arguments pre-filled. When called, the new function will use the pre-filled arguments along with any additional arguments provided.

        Args:
            *args: Variable length argument list to be partially applied to the function.

        Returns:
            A new [`StaticGenerativeFunction`][] with partially applied arguments.

        Example:
            ```python
            @gen
            def my_model(x, y):
                z = normal(x, 1.0) @ "z"
                return y * z


            partially_applied_model = my_model.partial_apply(2.0)
            # Now `partially_applied_model` is equivalent to a model that only takes 'y' as an argument
            ```
        """
        return gen(Pytree.partial(*args)(self.inline))


#############
# Decorator #
#############


def gen(f: Closure[R] | Callable[..., R]) -> StaticGenerativeFunction[R]:
    if isinstance(f, Closure):
        return StaticGenerativeFunction[R](f)
    else:
        closure = Pytree.partial()(f)
        return StaticGenerativeFunction[R](closure)


###########
# Exports #
###########

__all__ = [
    "AddressReuse",
    "StaticGenerativeFunction",
    "gen",
    "trace",
    "trace_p",
]
