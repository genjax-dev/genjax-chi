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
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.compiler.initial_style_primitive import (
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.compiler.interpreters.incremental import (
    Diff,
    incremental,
)
from genjax._src.core.compiler.interpreters.stateful import (
    StatefulHandler,
    stateful,
)
from genjax._src.core.compiler.staging import to_shape_fn
from genjax._src.core.generative import (
    GFI,
    Argdiffs,
    ChoiceMap,
    EditRequest,
    EmptyRequest,
    NotSupportedEditRequest,
    PrimitiveEditRequest,
    Regenerate,
    Retdiff,
    Score,
    Selection,
    StaticAddress,
    StaticAddressComponent,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.generative.choice_map import Address
from genjax._src.core.generative.interface import R
from genjax._src.core.pytree import Closure, Const, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    TypeAlias,
)

_WRAPPER_ASSIGNMENTS = (
    "__module__",
    "__name__",
    "__qualname__",
    "__doc__",
    "__annotations__",
)

#########
# Trace #
#########


@Pytree.dataclass
class FnTrace(Generic[R], Trace[R]):
    gen_fn: "Fn[R]"
    args: tuple[Any, ...]
    retval: R
    subtraces: dict[StaticAddress, Trace[R]]

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> R:
        return self.retval

    def get_gen_fn(self) -> GFI[R]:
        return self.gen_fn

    def get_choices(self) -> ChoiceMap:
        return ChoiceMap.d({
            address: subtrace.get_choices()
            for address, subtrace in self.subtraces.items()
        })

    def get_score(self) -> Score:
        return jnp.sum(
            jnp.array([tr.get_score() for tr in self.subtraces.values()], copy=False),
            axis=0,
        )

    def get_inner_trace(self, address: Address):
        if (
            isinstance(address, tuple)
            and len(address) == 1
            and cast(StaticAddress, address) not in self.subtraces
            and address[0] in self.subtraces
        ):
            warnings.warn(
                "use of get_subtrace(('x',)) is deprecated: prefer get_subtrace('x')",
                DeprecationWarning,
            )
            address = address[0]
        return self.subtraces[cast(StaticAddress, address)]


####################################
# Static (trie-like) edit request  #
####################################

StaticDict: TypeAlias = dict[StaticAddress, EditRequest]


@Pytree.dataclass(match_args=True)
class FnRequest(PrimitiveEditRequest):
    addressed: StaticDict


##############################
# Static language exceptions #
##############################


class AddressReuse(Exception):
    """Attempt to re-write an address in a GenJAX trace.

    Any given address for a random choice may only be written to once. You can choose a
    different name for the choice, or nest it into a scope where it is unique.
    """


class MissingAddress(Exception):
    """Attempt to assess a model without supplying values for all sampled addresses"""


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
    _: Const[StaticAddressComponent] | tuple[Const[StaticAddress], ...],
    gen_fn: GFI[R],
    args: tuple[Any, ...],
):
    return gen_fn.__abstract_call__(*args)


def trace(
    addr: StaticAddress,
    gen_fn: GFI[R],
    args: tuple[Any, ...],
):
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of `Fn`.

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
class DefHandler(StatefulHandler):
    def __init__(self):
        self.traces: dict[StaticAddress, Trace[Any]] = {}

    def record(self, addr, trace):
        if addr in self.traces:
            raise AddressReuse(addr)
        self.traces[addr] = trace

    @abstractmethod
    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GFI[R],
        args: tuple[Any, ...],
    ):
        pass

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


class SimulateHandler(DefHandler):
    def __init__(self):
        super().__init__()

    def yield_state(self):
        return self.traces

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GFI[Any],
        args: tuple[Any, ...],
    ):
        tr = gen_fn.simulate(args)
        self.record(addr, tr)
        v = tr.get_retval()
        return v


def simulate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(args):
        stateful_handler = SimulateHandler()
        retval = stateful(source_fn)(stateful_handler, *args)
        traces = stateful_handler.yield_state()
        return (args, retval, traces)

    return wrapper


##########
# Assess #
##########


@dataclass
class AssessHandler(DefHandler):
    def __init__(self, chm: ChoiceMap):
        super().__init__()
        self.choice_map_sample = chm
        self.score = jnp.zeros(())

    def yield_state(self):
        return (self.score,)

    def get_subsample(self, addr: StaticAddress) -> ChoiceMap:
        return self.choice_map_sample(addr)

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GFI[Any],
        args: tuple[Any, ...],
    ):
        submap = self.get_subsample(addr)
        if submap.static_is_empty():
            raise MissingAddress(addr)
        (score, v) = gen_fn.assess(submap, args)
        self.score += score
        return v


def assess_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(choice_map_sample: ChoiceMap, args):
        stateful_handler = AssessHandler(choice_map_sample)
        retval = stateful(source_fn)(stateful_handler, *args)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper


############################
# Generate request handler #
############################


@dataclass
class GenerateHandler(DefHandler):
    def __init__(
        self,
        choice_map: ChoiceMap,
    ):
        super().__init__()
        self.choice_map = choice_map
        self.weight: Weight = jnp.zeros(())

    def yield_state(
        self,
    ) -> tuple[Weight, dict[StaticAddress, Trace[Any]]]:
        return (
            self.weight,
            self.traces,
        )

    def get_subconstraint(
        self,
        addr: StaticAddress,
    ) -> ChoiceMap:
        return self.choice_map(addr)

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GFI[Any],
        args: tuple[Any, ...],
    ):
        subconstraint = self.get_subconstraint(addr)
        (tr, w) = gen_fn.generate(subconstraint, args)
        self.weight += w
        self.record(addr, tr)

        return tr.get_retval()


def generate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        choice_map: ChoiceMap,
        args: tuple[Any, ...],
    ):
        stateful_handler = GenerateHandler(choice_map)
        retval = stateful(source_fn)(stateful_handler, *args)
        (weight, traces) = stateful_handler.yield_state()
        return (
            weight,
            # Trace.
            (args, retval, traces),
        )

    return wrapper


###############
# Update edit #
###############


class UpdateHandler(DefHandler):
    def __init__(
        self,
        previous_trace: FnTrace[Any],
        constraint: ChoiceMap,
    ):
        super().__init__()
        self.previous_trace = previous_trace
        self.constraint = constraint
        self.weight = jnp.zeros(())
        self.bwd_constraints: list[ChoiceMap] = []

    def yield_state(self):
        return (
            self.weight,
            self.traces,
            self.bwd_constraints,
        )

    def get_subconstraint(self, addr: StaticAddress) -> ChoiceMap:
        return self.constraint(addr)

    def get_inner_trace(
        self,
        addr: StaticAddress,
    ):
        return self.previous_trace.get_inner_trace(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Diff))

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GFI[Any],
        args: tuple[Any, ...],
    ):
        argdiffs: Argdiffs = args
        subtrace = self.get_inner_trace(addr)
        constraint = self.get_subconstraint(addr)
        request = Update(constraint)
        (tr, w, retval_diff, bwd_request) = request.edit(
            subtrace,
            argdiffs,
        )
        assert isinstance(bwd_request, Update) and isinstance(
            bwd_request.constraint, ChoiceMap
        )
        self.bwd_constraints.append(bwd_request.constraint)
        self.weight += w
        self.record(addr, tr)

        return retval_diff


def update_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        previous_trace: FnTrace[R],
        constraint: ChoiceMap,
        diffs: tuple[Any, ...],
    ):
        stateful_handler = UpdateHandler(previous_trace, constraint)
        diff_primals = Diff.tree_primal(diffs)
        diff_tangents = Diff.tree_tangent(diffs)
        retval_diffs = incremental(source_fn)(
            stateful_handler, diff_primals, diff_tangents
        )
        retval_primals = Diff.tree_primal(retval_diffs)
        (
            weight,
            traces,
            bwd_requests,
        ) = stateful_handler.yield_state()
        return (
            (
                retval_diffs,
                weight,
                # Trace.
                (
                    diff_primals,
                    retval_primals,
                    traces,
                ),
                # Backward update problem.
                bwd_requests,
            ),
        )

    return wrapper


###################################
# Choice map edit request handler #
###################################


class StaticEditRequestHandler(DefHandler):
    def __init__(
        self,
        previous_trace: FnTrace[Any],
        addressed: StaticDict,
    ):
        super().__init__()
        self.previous_trace = previous_trace
        self.addressed = addressed
        self.weight = jnp.zeros(())
        self.bwd_requests: list[EditRequest] = []

    def yield_state(self):
        return (
            self.weight,
            self.traces,
            self.bwd_requests,
        )

    def get_subrequest(self, addr: StaticAddress) -> EditRequest:
        return self.addressed.get(addr, EmptyRequest())

    def get_subtrace(
        self,
        addr: StaticAddress,
    ):
        return self.previous_trace.get_subtrace(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Diff))

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GFI[Any],
        args: tuple[Any, ...],
    ):
        argdiffs: Argdiffs = args
        subtrace = self.get_subtrace(addr)
        subrequest = self.get_subrequest(addr)
        (tr, w, retval_diff, bwd_request) = subrequest.edit(
            subtrace,
            argdiffs,
        )
        self.bwd_requests.append(bwd_request)
        self.weight += w
        self.record(addr, tr)
        return retval_diff


def static_edit_request_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        previous_trace: FnTrace[R],
        addressed: dict[StaticAddress, EditRequest],
        diffs: tuple[Any, ...],
    ):
        stateful_handler = StaticEditRequestHandler(
            previous_trace,
            addressed,
        )
        diff_primals = Diff.tree_primal(diffs)
        diff_tangents = Diff.tree_tangent(diffs)
        retval_diffs = incremental(source_fn)(
            stateful_handler, diff_primals, diff_tangents
        )
        retval_primals = Diff.tree_primal(retval_diffs)
        (
            weight,
            traces,
            bwd_requests,
        ) = stateful_handler.yield_state()
        return (
            (
                retval_diffs,
                weight,
                # Trace.
                (
                    diff_primals,
                    retval_primals,
                    traces,
                ),
                # Backward update problem.
                bwd_requests,
            ),
        )

    return wrapper


#####################
# Select apply edit #
#####################


class RegenerateRequestHandler(DefHandler):
    def __init__(
        self,
        previous_trace: FnTrace[Any],
        selection: Selection,
        edit_request: EditRequest,
    ):
        super().__init__()
        self.previous_trace = previous_trace
        self.selection = selection
        self.edit_request = edit_request
        self.weight = jnp.zeros(())
        self.discard: list[ChoiceMap] = []

    def yield_state(self):
        return (
            self.weight,
            self.traces,
            self.discard,
        )

    def get_subselection(self, addr: StaticAddress) -> Selection:
        return self.selection(addr)

    def get_subtrace(
        self,
        addr: StaticAddress,
    ):
        return self.previous_trace.get_subtrace(addr)

    def handle_retval(self, v):
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Diff))

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GFI[Any],
        args: tuple[Any, ...],
    ):
        argdiffs: Argdiffs = args
        subtrace = self.get_subtrace(addr)
        subselection = self.get_subselection(addr)
        subrequest = Regenerate(subselection)
        tr, w, retval_diff, bwd_request = subrequest.edit(subtrace, argdiffs)
        assert isinstance(bwd_request, Update), bwd_request
        self.discard.append(bwd_request.constraint)
        self.weight += w
        self.record(addr, tr)

        return retval_diff


def regenerate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        previous_trace: FnTrace[R],
        selection: Selection,
        edit_request: EditRequest,
        diffs: tuple[Any, ...],
    ):
        stateful_handler = RegenerateRequestHandler(
            previous_trace,
            selection,
            edit_request,
        )
        diff_primals = Diff.tree_primal(diffs)
        diff_tangents = Diff.tree_tangent(diffs)
        retval_diffs = incremental(source_fn)(
            stateful_handler, diff_primals, diff_tangents
        )
        retval_primals = Diff.tree_primal(retval_diffs)
        (
            weight,
            traces,
            bwd_requests,
        ) = stateful_handler.yield_state()
        return (
            (
                retval_diffs,
                weight,
                # Trace.
                (
                    diff_primals,
                    retval_primals,
                    traces,
                ),
                # Backward update problem.
                bwd_requests,
            ),
        )

    return wrapper


#######################
# Generative function #
#######################


@Pytree.dataclass
class Fn(Generic[R], GFI[R]):
    """A `Fn` is a generative function relies on program
    transformations applied to JAX-compatible Python programs to implement the generative
    function interface.

    By virtue of the implementation, any source program which is provided to this generative function *must* be JAX traceable, meaning [all the footguns for programs that JAX exposes](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) apply to the source program.

    **Language restrictions**

    In addition to JAX footguns, there are a few more which are specific to the generative function interface semantics. Here is the full list of language restrictions (and capabilities):

    * One is allowed to use `jax.lax` control flow primitives _so long as the functions provided to the primitives do not contain `trace` invocations_. In other words, utilizing control flow primitives within the source of a `Fn`'s source program requires that the control flow primitives get *deterministic* computation.

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

    source_program: Closure[R]
    """
    The source program of the generative function. This is a JAX-compatible Python program.
    """

    def __get__(self, instance, _klass) -> "Fn[R]":
        """
        This method allows the @genjax.gen decorator to transform instance methods, turning them into `Fn[R]` calls.

        NOTE: if you assign an already-created `Fn` to a variable inside of a class, it will always receive the instance as its first method.
        """
        return self.partial_apply(instance) if instance else self

    # To get the type of return value, just invoke
    # the source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        return to_shape_fn(self.source_program, jnp.zeros)(*args)

    def __post_init__(self):
        wrapped = self.source_program.static_fn
        # Preserve the original function's docstring and name
        for k in _WRAPPER_ASSIGNMENTS:
            v = getattr(wrapped, k, None)
            if v is not None:
                object.__setattr__(self, k, v)

        object.__setattr__(self, "__wrapped__", wrapped)

    def handle_kwargs(self) -> "Fn[R]":
        @Pytree.partial()
        def kwarged_source(args, kwargs):
            return self.source_program(*args, **kwargs)

        return Fn(kwarged_source)

    def simulate(
        self,
        args: tuple[Any, ...],
    ) -> FnTrace[R]:
        (args, retval, traces) = simulate_transform(self.source_program)(args)
        return FnTrace(self, args, retval, traces)

    def generate(
        self,
        constraint: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[FnTrace[R], Weight]:
        (
            weight,
            # Trace.
            (
                args,
                retval,
                traces,
            ),
        ) = generate_transform(self.source_program)(constraint, args)
        return FnTrace(self, args, retval, traces), weight

    def project(
        self,
        trace: Trace[Any],
        selection: Selection,
    ) -> Weight:
        assert isinstance(trace, FnTrace)

        weight = jnp.array(0.0)
        for addr in trace.subtraces.keys():
            subprojection = selection(addr)
            subtrace = trace.get_subtrace(addr)
            weight += subtrace.project(subprojection)
        return weight

    def edit_update(
        self,
        trace: FnTrace[R],
        constraint: ChoiceMap,
        argdiffs: Argdiffs,
    ) -> tuple[FnTrace[R], Weight, Retdiff[R], EditRequest]:
        (
            (
                retval_diffs,
                weight,
                (
                    arg_primals,
                    retval_primals,
                    traces,
                ),
                bwd_requests,
            ),
        ) = update_transform(self.source_program)(trace, constraint, argdiffs)
        if not Diff.static_check_tree_diff(retval_diffs):
            retval_diffs = Diff.no_change(retval_diffs)

        def make_bwd_request(traces, subconstraints):
            addresses = traces.keys()
            chm = ChoiceMap.from_mapping(zip(addresses, subconstraints))
            return Update(chm)

        bwd_request = make_bwd_request(traces, bwd_requests)
        return (
            FnTrace(
                self,
                arg_primals,
                retval_primals,
                traces,
            ),
            weight,
            retval_diffs,
            bwd_request,
        )

    def edit_static_edit_request(
        self,
        trace: FnTrace[R],
        addressed: StaticDict,
        argdiffs: Argdiffs,
    ) -> tuple[FnTrace[R], Weight, Retdiff[R], EditRequest]:
        (
            (
                retval_diffs,
                weight,
                (
                    arg_primals,
                    retval_primals,
                    traces,
                ),
                bwd_requests,
            ),
        ) = static_edit_request_transform(self.source_program)(
            trace, addressed, argdiffs
        )

        def make_bwd_request(
            traces: dict[StaticAddress, Trace[R]],
            subrequests: list[EditRequest],
        ):
            return FnRequest(dict(zip(traces.keys(), subrequests)))

        bwd_request = make_bwd_request(traces, bwd_requests)
        return (
            FnTrace(
                self,
                arg_primals,
                retval_primals,
                traces,
            ),
            weight,
            retval_diffs,
            bwd_request,
        )

    def edit_regenerate(
        self,
        trace: FnTrace[R],
        selection: Selection,
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[FnTrace[R], Weight, Retdiff[R], EditRequest]:
        (
            (
                retval_diffs,
                weight,
                (
                    arg_primals,
                    retval_primals,
                    traces,
                ),
                discard,
            ),
        ) = regenerate_transform(self.source_program)(
            trace, selection, edit_request, argdiffs
        )

        def make_bwd_request(
            traces: dict[StaticAddress, Trace[R]],
            discard: list[ChoiceMap],
        ):
            return Update(ChoiceMap.from_mapping((zip(traces.keys(), discard))))

        bwd_request = make_bwd_request(traces, discard)
        return (
            FnTrace(
                self,
                arg_primals,
                retval_primals,
                traces,
            ),
            weight,
            retval_diffs,
            bwd_request,
        )

    def edit(
        self,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[FnTrace[R], Weight, Retdiff[R], EditRequest]:
        assert isinstance(trace, FnTrace)
        match edit_request:
            case Update(constraint):
                return self.edit_update(
                    trace,
                    constraint,
                    argdiffs,
                )

            case FnRequest(addressed):
                return self.edit_static_edit_request(
                    trace,
                    addressed,
                    argdiffs,
                )
            case Regenerate(selection):
                return self.edit_regenerate(
                    trace,
                    selection,
                    edit_request,
                    argdiffs,
                )
            case _:
                raise NotSupportedEditRequest(edit_request)

    def assess(
        self,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        (retval, score) = assess_transform(self.source_program)(chm, args)
        return (score, retval)

    def inline(self, *args):
        return self.source_program(*args)

    @property
    def partial_args(self) -> tuple[Any, ...]:
        """
        Returns the partially applied arguments of the generative function.

        This method retrieves the dynamically applied arguments that were used to create
        this Fn instance through partial application.

        Returns:
            tuple[Any, ...]: A tuple containing the partially applied arguments.

        Note:
            This method is particularly useful when working with partially applied
            generative functions, allowing access to the pre-filled arguments.
        """
        return self.source_program.dyn_args

    def partial_apply(self, *args) -> "Fn[R]":
        """
        Returns a new [`Fn`][] with the given arguments partially applied.

        This method creates a new [`Fn`][] that has some of its arguments pre-filled. When called, the new function will use the pre-filled arguments along with any additional arguments provided.

        Args:
            *args: Variable length argument list to be partially applied to the function.

        Returns:
            A new [`Fn`][] with partially applied arguments.

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
        all_args = self.source_program.dyn_args + args
        return gen(Closure[R](all_args, self.source_program.static_fn))


#############
# Decorator #
#############


def gen(f: Closure[R] | Callable[..., R]) -> Fn[R]:
    if isinstance(f, Closure):
        return Fn[R](f)
    else:
        closure = Closure[R]((), f)
        return gen(closure)


###########
# Exports #
###########

__all__ = [
    "AddressReuse",
    "Fn",
    "gen",
    "trace",
    "trace_p",
]
