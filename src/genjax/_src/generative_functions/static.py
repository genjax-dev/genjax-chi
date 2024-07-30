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
    Arguments,
    ChoiceMap,
    ChoiceMapBuilder,
    ChoiceMapConstraint,
    ChoiceMapEditRequest,
    ChoiceMapProjection,
    ChoiceMapSample,
    Constraint,
    EditRequest,
    GenerativeFunction,
    IncrementalChoiceMapEditRequest,
    Projection,
    Retdiff,
    Retval,
    Sample,
    SampleCoercableToChoiceMap,
    Score,
    Selection,
    SelectionProjection,
    SelectionRegenerateRequest,
    StaticAddress,
    StaticAddressComponent,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import (
    push_trace_overload_stack,
)
from genjax._src.core.interpreters.forward import (
    InitialStylePrimitive,
    StatefulHandler,
    forward,
    initial_style_bind,
)
from genjax._src.core.interpreters.incremental import Diff, incremental
from genjax._src.core.pytree import Closure, Const, Pytree
from genjax._src.core.typing import (
    Any,
    Bool,
    Callable,
    Generic,
    List,
    PRNGKey,
    TypeVar,
    overload,
    tuple,
)

A = TypeVar("A", bound=Arguments)
S = TypeVar("S", bound=Sample)
R = TypeVar("R", bound=Retval)
G = TypeVar("G", bound=GenerativeFunction)
C = TypeVar("C", bound=Constraint)
S = TypeVar("S", bound=Sample)
P = TypeVar("P", bound=Projection)
Tr = TypeVar("Tr", bound=Trace)
U = TypeVar("U", bound=EditRequest)


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
    SampleCoercableToChoiceMap,
    Trace["StaticGenerativeFunction", A, ChoiceMapSample, R],
):
    gen_fn: "StaticGenerativeFunction"
    arguments: A
    retval: R
    traced_addresses: AddressVisitor
    subtraces: List[Trace]
    score: Score
    cached_addresses: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cached_values: List[Any] = Pytree.field(default_factory=list)

    def get_args(self) -> A:
        return self.arguments

    def get_retval(self) -> R:
        return self.retval

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_cached_state(self, addr: StaticAddress) -> ChoiceMap:
        addresses = self.cached_addresses.get_visited()
        chm = ChoiceMap.empty()
        for addr, cached_val in zip(addresses, self.cached_values):
            chm = chm ^ ChoiceMapBuilder.a(addr, cached_val)
        return chm

    def get_sample(self) -> ChoiceMapSample:
        addresses = self.traced_addresses.get_visited()
        chm = ChoiceMap.empty()
        for addr, subtrace in zip(addresses, self.subtraces):
            chm = chm ^ ChoiceMapBuilder.a(addr, subtrace.get_sample())

        return ChoiceMapSample(chm)

    def get_choices(self) -> ChoiceMap:
        addresses = self.traced_addresses.get_visited()
        chm = ChoiceMap.empty()
        for addr, subtrace in zip(addresses, self.subtraces):
            assert isinstance(subtrace, SampleCoercableToChoiceMap)
            chm = chm ^ ChoiceMapBuilder.a(addr, subtrace.get_choices())

        return chm

    def get_score(self) -> Score:
        return self.score

    def get_subtrace(self, addr: StaticAddress):
        addresses = self.traced_addresses.get_visited()
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

# Deterministic caching intrinsic.
cache_p = InitialStylePrimitive("cache")


############################################################
# Trace call (denotes invocation of a generative function) #
############################################################


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(
    _: tuple[Const[StaticAddress], ...],
    gen_fn: GenerativeFunction[Tr, A, S, R, C, P, U],
    arguments: A,
):
    return gen_fn.__abstract_call__(*arguments)


def trace(
    addr: StaticAddress,
    gen_fn: GenerativeFunction[Tr, A, S, R, C, P, U],
    arguments: A,
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


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_call(
    _: tuple[Const[StaticAddress], ...],
    fn: Callable[[A], R],
    arguments: A,
) -> R:
    return fn(*arguments)


def cache(
    addr: StaticAddress,
    fn: Callable[[A], R],
    arguments: A,
) -> R:
    """Invoke a deterministic function and expose caching semantics to the
    current caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        fn: A function invoked as a callee of `StaticGenerativeFunction`.
        arguments: The arguments to the invocation.

    """
    addr = Pytree.tree_const(addr)
    return initial_style_bind(cache_p)(_abstract_call)(
        addr,
        fn,
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
class StaticHandler(StatefulHandler):
    @abstractmethod
    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction,
        arguments: Arguments,
    ):
        raise NotImplementedError

    def handle_cache(
        self,
        addr: StaticAddress,
        fn: Callable[[A], R],
        arguments: A,
    ) -> R:
        return fn(*arguments)

    def handle_retval(self, v):
        return jtu.tree_leaves(v)

    # By default, the interpreter handlers for this language
    # handle the two primitives we defined above
    # (`trace_p`, for random choices and `cache_p` for caching deterministic values)
    def handles(self, primitive):
        return primitive == trace_p or primitive == cache_p

    def dispatch(self, primitive, *tracers, **_params):
        in_tree = _params["in_tree"]
        num_consts = _params.get("num_consts", 0)
        non_const_tracers = tracers[num_consts:]
        if primitive == trace_p:
            addr, gen_fn, arguments = jtu.tree_unflatten(in_tree, non_const_tracers)
            addr = Pytree.tree_unwrap_const(addr)
            v = self.handle_trace(addr, gen_fn, arguments)
            return self.handle_retval(v)
        elif primitive == cache_p:
            addr, fn, arguments = jtu.tree_unflatten(in_tree, non_const_tracers)
            addr = Pytree.tree_unwrap_const(addr)
            v = self.handle_cache(addr, fn, arguments)
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
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    cache_addresses: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cached_values: List[Any] = Pytree.field(default_factory=list)

    def visit(self, addr: StaticAddress):
        self.address_visitor.visit(addr)

    def cache_visit(self, addr: StaticAddress):
        self.cache_addresses.visit(addr)

    def yield_state(
        self,
    ) -> tuple[AddressVisitor, List[Trace], Score, AddressVisitor, List[Any]]:
        return (
            self.address_visitor,
            self.address_traces,
            self.score,
            self.cache_addresses,
            self.cached_values,
        )

    def handle_cache(
        self,
        addr: StaticAddress,
        fn: Callable[[A], R],
        arguments: A,
    ) -> R:
        self.cache_visit(addr)
        r = fn(*arguments)
        self.cached_values.append(r)
        return r

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction,
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
            cache_addresses,
            cached_values,
        ) = stateful_handler.yield_state()
        return (
            (
                arguments,
                retval,
                address_visitor,
                address_traces,
                score,
            ),
            cache_addresses,
            cached_values,
        )

    return wrapper


##########
# Assess #
##########


@dataclass
class AssessHandler(StaticHandler):
    key: PRNGKey
    sample: ChoiceMapSample
    score: Score = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cache_addresses: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def visit(self, addr: StaticAddress):
        self.address_visitor.visit(addr)

    def cache_visit(self, addr: StaticAddress):
        self.cache_addresses.visit(addr)

    def yield_state(self) -> tuple[Score]:
        return (self.score,)

    def get_subsample(self, addr: StaticAddress) -> ChoiceMapSample:
        return self.sample(addr)

    def handle_cache(
        self,
        addr: StaticAddress,
        fn: Callable[[A], R],
        arguments: A,
    ) -> R:
        self.cache_visit(addr)
        r = fn(*arguments)
        return r

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction,
        arguments: Arguments,
    ):
        self.visit(addr)
        submap = self.get_subsample(addr)
        self.key, sub_key = jax.random.split(self.key)
        (score, v) = gen_fn.assess(sub_key, submap, arguments)
        self.score += score
        return v


def assess_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key: PRNGKey, constraints, arguments):
        stateful_handler = AssessHandler(key, constraints)
        retval = forward(source_fn)(stateful_handler, *arguments)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper


#####################
# Importance update #
#####################


@dataclass
class ChoiceMapImportanceEditHandler(StaticHandler):
    key: PRNGKey
    choice_map_constraint: ChoiceMapConstraint
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: Score = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: Weight = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_projections: List[Projection] = Pytree.field(default_factory=list)
    cache_addresses: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cached_values: List[Any] = Pytree.field(default_factory=list)

    def visit(self, addr: StaticAddress):
        self.address_visitor.visit(addr)

    def cache_visit(self, addr: StaticAddress):
        self.cache_addresses.visit(addr)

    def yield_state(
        self,
    ) -> tuple[
        Score,
        Weight,
        AddressVisitor,
        List[Trace],
        List[Projection],
        AddressVisitor,
        List[Any],
    ]:
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_projections,
            self.cache_addresses,
            self.cached_values,
        )

    def get_subconstraint(
        self,
        addr: StaticAddress,
    ) -> Constraint:
        return self.choice_map_constraint(addr)

    def handle_cache(
        self,
        addr: StaticAddress,
        fn: Callable[[A], R],
        arguments: A,
    ) -> R:
        self.cache_visit(addr)
        r = fn(*arguments)
        self.cached_values.append(r)
        return r

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction,
        arguments: Arguments,
    ):
        self.visit(addr)
        subconstraint = self.get_subconstraint(addr)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w, bwd_projection) = gen_fn.importance_edit(
            sub_key, subconstraint, arguments
        )
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_projections.append(bwd_projection)

        return tr.get_retval()


def choice_map_importance_edit_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        key: PRNGKey,
        choice_map_constraint: ChoiceMapConstraint,
        arguments: Arguments,
    ):
        stateful_handler = ChoiceMapImportanceEditHandler(key, choice_map_constraint)
        retval = forward(source_fn)(stateful_handler, *arguments)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_projections,
            cache_addresses,
            cached_values,
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
            cache_addresses,
            cached_values,
        )

    return wrapper


###################
# Choice map edit #
###################


@dataclass
class ChoiceMapEditRequestHandler(StaticHandler):
    key: PRNGKey
    previous_trace: StaticTrace
    choice_map_constraint: ChoiceMapConstraint
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: Score = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: Weight = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_discard_constraints: List[ChoiceMap] = Pytree.field(default_factory=list)
    cache_addresses: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cached_values: List[Any] = Pytree.field(default_factory=list)

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def cache_visit(self, addr: StaticAddress):
        self.cache_addresses.visit(addr)

    def yield_state(
        self,
    ) -> tuple[
        Score,
        Weight,
        AddressVisitor,
        List[Trace],
        List[ChoiceMap],
        AddressVisitor,
        List[Any],
    ]:
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_discard_constraints,
            self.cache_addresses,
            self.cached_values,
        )

    def get_subtrace(
        self,
        sub_gen_fn: GenerativeFunction,
        addr: StaticAddress,
    ):
        return self.previous_trace.get_subtrace(addr)

    def get_subconstraint(
        self,
        addr: StaticAddress,
    ) -> ChoiceMapConstraint:
        return self.choice_map_constraint(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v)

    def handle_cache(
        self,
        addr: StaticAddress,
        fn: Callable[[A], R],
        arguments: A,
    ) -> R:
        self.cache_visit(addr)
        r = fn(*arguments)
        self.cached_values.append(r)
        return r

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction,
        arguments: Arguments,
    ):
        self.visit(addr)
        subtrace = self.get_subtrace(gen_fn, addr)
        subconstraint = self.get_subconstraint(addr)
        self.key, sub_key = jax.random.split(self.key)
        request = ChoiceMapEditRequest(subconstraint)
        (tr, w, _, bwd_request) = request.edit(sub_key, subtrace, arguments)
        discard = bwd_request.constraint.choice_map
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_discard_constraints.append(discard)

        return tr.get_retval()


def choice_map_edit_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, previous_trace, constraints, arguments: tuple):
        stateful_handler = ChoiceMapEditRequestHandler(key, previous_trace, constraints)
        retval = forward(source_fn)(stateful_handler, *arguments)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_discards,
            cache_addresses,
            cached_values,
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
            cache_addresses,
            cached_values,
        )

    return wrapper


@dataclass
class IncrementalChoiceMapEditRequestHandler(StaticHandler):
    key: PRNGKey
    previous_trace: StaticTrace
    choice_map_constraint: ChoiceMapConstraint
    propagate_incremental: Bool = Pytree.static()
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: Score = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: Weight = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_discard_constraints: List[ChoiceMap] = Pytree.field(default_factory=list)
    cache_addresses: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cached_values: List[Any] = Pytree.field(default_factory=list)

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def cache_visit(self, addr: StaticAddress):
        self.cache_addresses.visit(addr)

    def yield_state(
        self,
    ) -> tuple[
        Score,
        Weight,
        AddressVisitor,
        List[Trace],
        List[ChoiceMap],
        AddressVisitor,
        List[Any],
    ]:
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_discard_constraints,
            self.cache_addresses,
            self.cached_values,
        )

    def get_subtrace(
        self,
        sub_gen_fn: GenerativeFunction,
        addr: StaticAddress,
    ):
        return self.previous_trace.get_subtrace(addr)

    def get_subconstraint(
        self,
        addr: StaticAddress,
    ) -> ChoiceMapConstraint:
        return self.choice_map_constraint(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v)

    def handle_cache(
        self,
        addr: StaticAddress,
        fn: Callable[[A], R],
        arguments: A,
    ) -> R:
        self.cache_visit(addr)
        r = fn(*arguments)
        self.cached_values.append(r)
        return r

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction,
        arguments: Arguments,
    ):
        self.visit(addr)
        subtrace = self.get_subtrace(gen_fn, addr)
        subconstraint = self.get_subconstraint(addr)
        self.key, sub_key = jax.random.split(self.key)
        if self.propagate_incremental:
            request = IncrementalChoiceMapEditRequest(subconstraint)
        else:
            request = ChoiceMapEditRequest(subconstraint)
        (tr, w, _, bwd_request) = request.edit(sub_key, subtrace, arguments)
        discard = bwd_request.constraint.choice_map
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_discard_constraints.append(discard)

        return tr.get_retval()


def incremental_choice_map_edit_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        key: PRNGKey,
        previous_trace: StaticTrace,
        constraint: ChoiceMapConstraint,
        argdiffs: Argdiffs,
        propagate_incremental: Bool,
    ):
        stateful_handler = IncrementalChoiceMapEditRequestHandler(
            key,
            previous_trace,
            constraint,
            propagate_incremental,
        )
        diff_primals = Diff.primal(argdiffs)
        diff_tangents = Diff.tangent(argdiffs)
        retdiff = incremental(source_fn)(
            stateful_handler,
            diff_primals,
            diff_tangents,
        )
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_discards,
            cache_addresses,
            cached_values,
        ) = stateful_handler.yield_state()
        retval = Diff.primal(retdiff)
        return (
            (
                weight,
                # Trace.
                (
                    diff_primals,
                    retval,
                    address_visitor,
                    address_traces,
                    score,
                ),
                # Retdiff.
                retdiff,
                # Backward update problem.
                bwd_discards,
            ),
            cache_addresses,
            cached_values,
        )

    return wrapper


#############################
# Selection regenerate edit #
#############################


@dataclass
class SelectionRegenerateEditHandler(StaticHandler):
    key: PRNGKey
    previous_trace: StaticTrace
    selection: Selection
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: Score = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: Weight = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_discards: List[ChoiceMap] = Pytree.field(default_factory=list)
    cache_addresses: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cached_values: List[Any] = Pytree.field(default_factory=list)

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def cache_visit(self, addr: StaticAddress):
        self.cache_addresses.visit(addr)

    def yield_state(
        self,
    ) -> tuple[
        Score,
        Weight,
        AddressVisitor,
        List[Trace],
        List[ChoiceMap],
        AddressVisitor,
        List[Any],
    ]:
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_discards,
            self.cache_addresses,
            self.cached_values,
        )

    def get_subtrace(
        self,
        sub_gen_fn: GenerativeFunction,
        addr: StaticAddress,
    ):
        return self.previous_trace.get_subtrace(addr)

    def get_subselection(
        self,
        addr: StaticAddress,
    ) -> Selection:
        return self.selection(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v)

    def handle_cache(
        self,
        addr: StaticAddress,
        fn: Callable[[A], R],
        arguments: A,
    ) -> R:
        self.cache_visit(addr)
        r = fn(*arguments)
        self.cached_values.append(r)
        return r

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction,
        arguments: Arguments,
    ):
        self.visit(addr)
        subtrace = self.get_subtrace(gen_fn, addr)
        subselection = self.get_subselection(addr)
        self.key, sub_key = jax.random.split(self.key)
        request = SelectionRegenerateRequest(subselection)
        (tr, w, _, bwd_request) = request.edit(sub_key, subtrace, arguments)
        discard = bwd_request.constraint.choice_map
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_discards.append(discard)

        return tr.get_retval()


def selection_regenerate_edit_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        key: PRNGKey,
        previous_trace: StaticTrace,
        selection: Selection,
        arguments: Arguments,
    ):
        stateful_handler = SelectionRegenerateEditHandler(
            key, previous_trace, selection
        )
        retval = forward(source_fn)(stateful_handler, *arguments)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_discards,
            cache_addresses,
            cached_values,
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
            cache_addresses,
            cached_values,
        )

    return wrapper


#######################
# Generative function #
#######################


# Callee syntactic sugar handler.
def handler_trace_with_static(
    addr: StaticAddressComponent | StaticAddress,
    gen_fn: GenerativeFunction,
    arguments: Arguments,
):
    return trace(addr if isinstance(addr, tuple) else (addr,), gen_fn, arguments)


@Pytree.dataclass
class StaticGenerativeFunction(
    Generic[A, R],
    GenerativeFunction[
        StaticTrace[A, R],
        A,
        ChoiceMapSample,
        R,
        ChoiceMapConstraint,
        ChoiceMapProjection | SelectionProjection,
        ChoiceMapEditRequest[A]
        | IncrementalChoiceMapEditRequest[A]
        | SelectionRegenerateRequest[A],
    ],
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
        (
            (arguments, retval, address_visitor, address_traces, score),
            cache_addresses,
            cached_values,
        ) = simulate_transform(syntax_sugar_handled)(key, arguments)
        return StaticTrace(
            self,
            arguments,
            retval,
            address_visitor,
            address_traces,
            score,
            cache_addresses,
            cached_values,
        )

    def assess(
        self,
        key: PRNGKey,
        sample: ChoiceMap | ChoiceMapSample,
        arguments: A,
    ) -> tuple[Score, R]:
        sample = (
            sample if isinstance(sample, ChoiceMapSample) else ChoiceMapSample(sample)
        )
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (retval, score) = assess_transform(syntax_sugar_handled)(key, sample, arguments)
        return (score, retval)

    def importance_edit(
        self,
        key: PRNGKey,
        constraint: ChoiceMapConstraint,
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, ChoiceMapProjection]:
        match constraint:
            case ChoiceMapConstraint():
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
                    cache_addresses,
                    cached_values,
                ) = choice_map_importance_edit_transform(syntax_sugar_handled)(
                    key, constraint, arguments
                )

                def make_bwd_proj(visitor, subrequests) -> ChoiceMapProjection:
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
                        cache_addresses,
                        cached_values,
                    ),
                    weight,
                    bwd_proj,
                )

    def project_edit(
        self,
        key: PRNGKey,
        trace: Trace,
        projection: ChoiceMapProjection | SelectionProjection,
    ) -> tuple[Weight, ChoiceMapConstraint]:
        raise NotImplementedError

    def choice_map_edit(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        constraint: ChoiceMapConstraint,
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, ChoiceMapConstraint]:
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
            cache_addresses,
            cached_values,
        ) = choice_map_edit_transform(syntax_sugar_handled)(
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
                cache_addresses,
                cached_values,
            ),
            weight,
            bwd_discard,
        )

    def incremental_choice_map_edit(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        constraint: ChoiceMapConstraint,
        argdiffs: Argdiffs,
        propagate_incremental: Bool,
    ) -> tuple[StaticTrace[A, R], Weight, Retdiff, ChoiceMapConstraint]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (
            (
                weight,
                (
                    arguments,
                    retval,
                    address_visitor,
                    address_traces,
                    score,
                ),
                retdiff,
                bwd_discard_constraints,
            ),
            cache_addresses,
            cached_values,
        ) = incremental_choice_map_edit_transform(syntax_sugar_handled)(
            key, trace, constraint, argdiffs, propagate_incremental
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
                cache_addresses,
                cached_values,
            ),
            weight,
            retdiff,
            bwd_discard,
        )

    def selection_regenerate_edit(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        selection: Selection,
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, ChoiceMapConstraint]:
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
            cache_addresses,
            cached_values,
        ) = selection_regenerate_edit_transform(syntax_sugar_handled)(
            key, trace, selection, arguments
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
                cache_addresses,
                cached_values,
            ),
            weight,
            bwd_discard,
        )

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        request: ChoiceMapEditRequest[A],
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, Retdiff, ChoiceMapEditRequest[A]]:
        pass

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        request: IncrementalChoiceMapEditRequest[A],
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, Retdiff, IncrementalChoiceMapEditRequest[A]]:
        pass

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        request: SelectionRegenerateRequest[A],
        arguments: A,
    ) -> tuple[StaticTrace[A, R], Weight, Retdiff, ChoiceMapEditRequest[A]]:
        pass

    def edit(
        self,
        key: PRNGKey,
        trace: StaticTrace[A, R],
        request: ChoiceMapEditRequest[A]
        | IncrementalChoiceMapEditRequest[A]
        | SelectionRegenerateRequest[A],
        arguments: A,
    ) -> tuple[
        StaticTrace[A, R],
        Weight,
        Retdiff,
        ChoiceMapEditRequest[A] | IncrementalChoiceMapEditRequest[A],
    ]:
        match request:
            case ChoiceMapEditRequest(choice_map_constraint):
                new_trace, weight, bwd_constraint = self.choice_map_edit(
                    key, trace, choice_map_constraint, arguments
                )
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    ChoiceMapEditRequest(bwd_constraint),
                )

            case IncrementalChoiceMapEditRequest(choice_map_constraint):
                new_trace, weight, retdiff, bwd_constraint = (
                    self.incremental_choice_map_edit(
                        key, trace, choice_map_constraint, arguments
                    )
                )
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    IncrementalChoiceMapEditRequest(bwd_constraint),
                )

            case SelectionRegenerateRequest(selection):
                new_trace, weight, bwd_constraint = self.selection_regenerate_edit(
                    key, trace, selection, arguments
                )
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    ChoiceMapEditRequest(bwd_constraint),
                )

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
