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
    Arguments,
    Assessable,
    ChoiceMap,
    ChoiceMapBuilder,
    ChoiceMapCoercable,
    ChoiceMapConstraint,
    ChoiceMapProjection,
    ChoiceMapSample,
    Constraint,
    EmptyTrace,
    GeneralRegenerateRequest,
    GeneralUpdateRequest,
    GenerativeFunction,
    ImportanceRequest,
    Projection,
    ProjectRequest,
    Retval,
    Sample,
    Score,
    SelectionProjection,
    Simulateable,
    StaticAddress,
    StaticAddressComponent,
    Trace,
    UpdateRequest,
    Weight,
)
from genjax._src.core.generative.core import push_trace_overload_stack
from genjax._src.core.interpreters.forward import (
    InitialStylePrimitive,
    StatefulHandler,
    forward,
    initial_style_bind,
)
from genjax._src.core.pytree import Closure, Const, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    List,
    PRNGKey,
    TypeVar,
    tuple,
)

A = TypeVar("A", bound=Arguments)
S = TypeVar("S", bound=Sample)
R = TypeVar("R", bound=Retval)
G = TypeVar("G", bound=GenerativeFunction)


# Usage in transforms: checks for duplicate addresses.
@Pytree.dataclass
class AddressVisitor(Pytree):
    visited: List[StaticAddress] = Pytree.static(default_factory=list)

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
class StaticTrace(
    Generic[A, R],
    ChoiceMapCoercable,
    Trace["StaticGenerativeFunction", A, ChoiceMapSample, R],
):
    gen_fn: "StaticGenerativeFunction"
    arguments: A
    retval: R
    addresses: AddressVisitor
    subtraces: List[Trace]
    score: Score

    def get_args(self) -> A:
        return self.arguments

    def get_retval(self) -> R:
        return self.retval

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_sample(self) -> ChoiceMapSample:
        addresses = self.addresses.get_visited()
        chm = ChoiceMap.empty()
        for addr, subtrace in zip(addresses, self.subtraces):
            chm = chm ^ ChoiceMapBuilder.a(addr, subtrace.get_sample())

        return ChoiceMapSample(chm)

    def get_choices(self) -> ChoiceMap:
        addresses = self.addresses.get_visited()
        chm = ChoiceMap.empty()
        for addr, subtrace in zip(addresses, self.subtraces):
            assert isinstance(subtrace, ChoiceMapCoercable)
            chm = chm ^ ChoiceMapBuilder.a(addr, subtrace.get_choices())

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

    Any given address for a random choice may only be written to once. You can
    choose a different name for the choice, or nest it into a scope where it is
    unique.

    """


##############
# Primitives #
##############

# Generative function trace intrinsic.
trace_p = InitialStylePrimitive("trace")


############################################################
# Trace call (denotes invocation of a generative function) #
############################################################

Tr = TypeVar("Tr", bound=Trace)


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(
    _: tuple[Const[StaticAddress], ...],
    gen_fn: Simulateable[Tr, A, S, R],
    arguments: A,
):
    return gen_fn.__abstract_call__(*arguments)


def trace(
    addr: StaticAddress,
    gen_fn: GenerativeFunction,
    arguments: tuple,
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
        arguments,
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
class StaticHandler(Generic[G], StatefulHandler):
    @abstractmethod
    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: G,
        arguments: Arguments,
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
        addr, gen_fn, arguments = jtu.tree_unflatten(in_tree, non_const_tracers)
        addr = Pytree.tree_unwrap_const(addr)
        if primitive == trace_p:
            v = self.handle_trace(addr, gen_fn, arguments)
            return self.handle_retval(v)
        else:
            raise Exception("Illegal primitive: {}".format(primitive))


############
# Simulate #
############


@dataclass
class SimulateHandler(StaticHandler[Simulateable]):
    key: PRNGKey
    score: Score = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    address_traces: List[Trace] = Pytree.field(default_factory=list)

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
        gen_fn: Simulateable,
        arguments: Arguments,
    ):
        self.visit(addr)
        self.key, sub_key = jax.random.split(self.key)
        tr = gen_fn.simulate(sub_key, arguments)
        score = tr.get_score()
        self.address_traces.append(tr)
        self.score += score
        v = tr.get_retval()
        return v


def simulate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, arguments):
        stateful_handler = SimulateHandler(key)
        retval = forward(source_fn)(stateful_handler, *arguments)
        (
            address_visitor,
            address_traces,
            score,
        ) = stateful_handler.yield_state()
        return (
            arguments,
            retval,
            address_visitor,
            address_traces,
            score,
        )

    return wrapper


#####################
# Importance update #
#####################


@dataclass
class ImportanceHandler(
    StaticHandler[ImportanceRequest.SupportsImportance],
):
    key: PRNGKey
    choice_map_constraint: ChoiceMapConstraint
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_projections: List[Projection] = Pytree.field(default_factory=list)

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_projections,
        )

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def get_subconstraint(
        self,
        addr: StaticAddress,
    ) -> Constraint:
        return self.choice_map_constraint(addr)

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: ImportanceRequest.SupportsImportance,
        arguments: tuple,
    ):
        self.visit(addr)
        subconstraint = self.get_subconstraint(addr)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w, bwd_projection) = gen_fn.importance_update(
            sub_key, subconstraint, arguments
        )
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_projections.append(bwd_projection)

        return tr.get_retval()


def importance_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, arguments: tuple):
        stateful_handler = ImportanceHandler(key, constraints)
        retval = forward(source_fn)(stateful_handler, *arguments)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_projections,
        ) = stateful_handler.yield_state()
        return (
            (
                weight,
                # Trace.
                (
                    arguments,
                    retval,
                    address_visitor,
                    address_traces,
                    score,
                ),
                # Backward update problem.
                bwd_projections,
            ),
        )

    return wrapper


##################
# General update #
##################


@dataclass
class GeneralUpdateHandler(
    StaticHandler[GeneralUpdateRequest.SupportsGeneralUpdate],
):
    key: PRNGKey
    previous_trace: StaticTrace
    choice_map_constraint: ChoiceMapConstraint
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_discard_constraints: List[Constraint] = Pytree.field(default_factory=list)

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_discard_constraints,
        )

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def get_subtrace(
        self,
        sub_gen_fn: GenerativeFunction,
        addr: StaticAddress,
    ):
        return self.previous_trace.get_subtrace(addr)

    def get_subconstraint(
        self,
        addr: StaticAddress,
    ) -> Constraint:
        return self.choice_map_constraint(addr)

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GeneralUpdateRequest.SupportsGeneralUpdate,
        arguments: tuple,
    ):
        self.visit(addr)
        subtrace = self.get_subtrace(gen_fn, addr)
        subconstraint = self.get_subconstraint(addr)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w, bwd_discard_constraint) = gen_fn.general_update(
            sub_key, subtrace, subconstraint, arguments
        )
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_discard_constraints.append(bwd_discard_constraint)

        return tr.get_retval()


def general_update_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, previous_trace, constraints, arguments: tuple):
        stateful_handler = GeneralUpdateHandler(key, previous_trace, constraints)
        retval = forward(source_fn)(stateful_handler, *arguments)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_discards,
        ) = stateful_handler.yield_state()
        return (
            (
                weight,
                # Trace.
                (
                    arguments,
                    retval,
                    address_visitor,
                    address_traces,
                    score,
                ),
                # Backward update problem.
                bwd_discards,
            ),
        )

    return wrapper


##########
# Assess #
##########


@dataclass
class AssessHandler(StaticHandler):
    key: PRNGKey
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
        gen_fn: Assessable,
        arguments: tuple,
    ):
        submap = self.get_subsample(addr)
        self.key, sub_key = jax.random.split(self.key)
        (score, v) = gen_fn.assess(sub_key, submap, arguments)
        self.score += score
        return v


def assess_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, arguments):
        stateful_handler = AssessHandler(key, constraints)
        retval = forward(source_fn)(stateful_handler, *arguments)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper


#######################
# Generative function #
#######################


# Callee syntactic sugar handler.
def handler_trace_with_static(
    addr: StaticAddressComponent | StaticAddress,
    gen_fn: GenerativeFunction,
    arguments: tuple,
):
    return trace(addr if isinstance(addr, tuple) else (addr,), gen_fn, arguments)


SupportedGeneralConstraints = ChoiceMapConstraint
SupportedImportanceConstraints = ChoiceMapConstraint
SupportedProjections = SelectionProjection


@Pytree.dataclass
class StaticGenerativeFunction(
    Generic[A, R],
    Simulateable[StaticTrace[A, R], A, ChoiceMapSample, R],
    Assessable[StaticTrace[A, R], A, ChoiceMapSample, R],
    ImportanceRequest[
        EmptyTrace["StaticGenerativeFunction"], StaticTrace[A, R]
    ].SupportsImportance[
        StaticTrace[A, R],
        SupportedImportanceConstraints,
        A,
        ChoiceMapSample,
        R,
    ],
    GeneralUpdateRequest[StaticTrace[A, R],].UseAsDefaultUpdate[
        StaticTrace[A, R],
        SupportedGeneralConstraints,
        SupportedGeneralConstraints,
        A,
        ChoiceMapSample,
        R,
    ],
    GeneralRegenerateRequest[
        StaticTrace[A, R],
        StaticTrace[A, R],
    ].SupportsGeneralRegenerate[
        StaticTrace[A, R],
        StaticTrace[A, R],
        A,
        ChoiceMapSample,
        R,
        SupportedProjections,
    ],
    ProjectRequest.SupportsProject[
        StaticTrace[A, R],
        A,
        ChoiceMapSample,
        R,
        SupportedProjections,
    ],
    GenerativeFunction[StaticTrace[A, R], A, ChoiceMapSample, R],
):
    """A `StaticGenerativeFunction` is a generative function which relies on
    program transformations applied to JAX-compatible Python programs to
    implement the generative function interface.

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

    source: Closure
    """The source program of the generative function.

    This is a JAX-compatible Python program.

    """

    # To get the type of return value, just invoke
    # the source (with abstract tracer arguments).
    def __abstract_call__(self, *arguments) -> Any:
        return self.source(*arguments)

    def handle_kwargs(self) -> GenerativeFunction:
        @Pytree.partial()
        def kwarged_source(arguments, kwargs):
            return self.source(*arguments, **kwargs)

        return StaticGenerativeFunction(kwarged_source)

    def simulate(
        self,
        key: PRNGKey,
        arguments: A,
    ) -> StaticTrace[A, R]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (arguments, retval, address_visitor, address_traces, score) = (
            simulate_transform(syntax_sugar_handled)(key, arguments)
        )
        return StaticTrace(
            self,
            arguments,
            retval,
            address_visitor,
            address_traces,
            score,
        )

    def importance_update(
        self,
        key: PRNGKey,
        constraint: SupportedImportanceConstraints,
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, Projection]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (
            (
                weight,
                (
                    _,
                    retval,
                    address_visitor,
                    address_traces,
                    score,
                ),
                projections,
            ),
        ) = importance_transform(syntax_sugar_handled)(key, constraint, arguments)

        def make_bwd_proj(visitor, subrequests) -> Projection:
            addresses = visitor.get_visited()
            addresses = Pytree.tree_unwrap_const(addresses)
            chm = ChoiceMap.empty()
            for addr, subprojection in zip(addresses, projections):
                chm = chm ^ ChoiceMapBuilder.a(addr, subprojection)
            return ChoiceMapProjection(chm)

        bwd_proj = make_bwd_proj(address_visitor, projections)
        return (
            StaticTrace(
                self,
                arguments,
                retval,
                address_visitor,
                address_traces,
                score,
            ),
            weight,
            bwd_proj,
        )

    def general_update(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        constraint: SupportedGeneralConstraints,
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, SupportedGeneralConstraints]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (
            (
                weight,
                (
                    _,
                    retval,
                    address_visitor,
                    address_traces,
                    score,
                ),
                bwd_discard_constraints,
            ),
        ) = general_update_transform(syntax_sugar_handled)(
            key, trace, constraint, arguments
        )

        def make_bwd_discard(visitor, subdiscards) -> ChoiceMapConstraint:
            addresses = visitor.get_visited()
            addresses = Pytree.tree_unwrap_const(addresses)
            chm = ChoiceMap.empty()
            for addr, subdiscard in zip(addresses, subdiscards):
                chm = chm ^ ChoiceMapBuilder.a(addr, subdiscard)
            return ChoiceMapConstraint(chm)

        bwd_discard = make_bwd_discard(address_visitor, bwd_discard_constraints)
        return (
            StaticTrace(
                self,
                arguments,
                retval,
                address_visitor,
                address_traces,
                score,
            ),
            weight,
            bwd_discard,
        )

    def general_regenerate(
        self,
        key: PRNGKey,
        trace: Trace,
        projection: SupportedProjections,
        arguments: Arguments,
    ) -> tuple[Trace, Weight, Sample]:
        raise NotImplementedError

    def project_update(
        self,
        key: PRNGKey,
        trace: Trace,
        projection: SupportedProjections,
    ) -> tuple[Weight, UpdateRequest]:
        raise NotImplementedError

    def assess(
        self,
        key: PRNGKey,
        sample: ChoiceMap | ChoiceMapSample,
        arguments: tuple,
    ) -> tuple[Score, Retval]:
        sample = (
            sample if isinstance(sample, ChoiceMapSample) else ChoiceMapSample(sample)
        )
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (retval, score) = assess_transform(syntax_sugar_handled)(key, sample, arguments)
        return (score, retval)

    def inline(self, *arguments):
        return self.source(*arguments)


#############
# Decorator #
#############


def gen(f: Callable[..., Any]) -> StaticGenerativeFunction:
    if isinstance(f, Closure):
        return StaticGenerativeFunction(f)
    else:
        closure = Pytree.partial()(f)
        return StaticGenerativeFunction(closure)


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
