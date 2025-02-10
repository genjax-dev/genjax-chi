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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    ChoiceMapConstraint,
    Constraint,
    EditRequest,
    EmptyRequest,
    GenerativeFunction,
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
from genjax._src.core.generative.generative_function import R, push_trace_overload_stack
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
from genjax._src.core.interpreters.staging import to_shape_fn
from genjax._src.core.pytree import Closure, Const, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    PRNGKey,
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
class StaticTrace(Generic[R], Trace[R]):
    gen_fn: "StaticGenerativeFunction[R]"
    args: tuple[Any, ...]
    retval: R
    subtraces: dict[StaticAddress, Trace[R]]

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> R:
        return self.retval

    def get_gen_fn(self) -> GenerativeFunction[R]:
        return self.gen_fn

    def get_choices(self) -> ChoiceMap:
        return ChoiceMap.d({
            address: subtrace.get_choices()
            for address, subtrace in self.subtraces.items()
        })

    def get_score(self) -> Score:
        return jnp.sum(
            jnp.array([tr.get_score() for tr in self.subtraces.values()], copy=False),
        )

    def get_subtrace(self, addr: StaticAddress):
        if (
            isinstance(addr, tuple)
            and len(addr) == 1
            and addr not in self.subtraces
            and addr[0] in self.subtraces
        ):
            warnings.warn(
                "use of get_subtrace(('x',)) is deprecated: prefer get_subtrace('x')",
                DeprecationWarning,
            )
            addr = addr[0]

        return self.subtraces[addr]


@Pytree.dataclass
class VoidTrace(Trace[None]):
    """Under normal circumstances, you should not use or receive an object
    of this type, which represents a Trace whose content is not yet known."""

    def get_retval(self):
        return None

    def get_args(self):
        return ()

    def get_score(self) -> Score:
        return jnp.zeros(())

    def get_choices(self):
        return ChoiceMap.empty()

    def get_gen_fn(self):
        return StaticGenerativeFunction(Closure((), lambda: None))


VOID_TRACE = VoidTrace()

####################################
# Static (trie-like) edit request  #
####################################

StaticDict: TypeAlias = dict[StaticAddressComponent | StaticAddress, EditRequest]


@Pytree.dataclass(match_args=True)
class StaticRequest(PrimitiveEditRequest):
    addressed: StaticDict


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
    _: Const[StaticAddressComponent] | tuple[Const[StaticAddress], ...],
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
class StaticHandler(StatefulHandler):
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
        gen_fn: GenerativeFunction[R],
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


class SimulateHandler(StaticHandler):
    def __init__(self, key: PRNGKey):
        super().__init__()
        self.key = key
        self.key_counter = 1

    def fresh_key_and_increment(self):
        new_key = jax.random.fold_in(self.key, self.key_counter)
        self.key_counter += 1
        return new_key

    def yield_state(self):
        return self.traces

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        sub_key = self.fresh_key_and_increment()
        tr = gen_fn.simulate(sub_key, args)
        self.record(addr, tr)
        v = tr.get_retval()
        return v


def simulate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, args):
        stateful_handler = SimulateHandler(key)
        retval = forward(source_fn)(stateful_handler, *args)
        traces = stateful_handler.yield_state()
        return (args, retval, traces)

    return wrapper


##########
# Assess #
##########


@dataclass
class AssessHandler(StaticHandler):
    def __init__(self, choice_map_sample: ChoiceMap):
        super().__init__()
        self.choice_map_sample = choice_map_sample
        self.score = jnp.zeros(())

    def yield_state(self):
        return (self.score,)

    def get_subsample(self, addr: StaticAddress) -> ChoiceMap:
        return self.choice_map_sample(addr)

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
    def wrapper(choice_map_sample: ChoiceMap, args):
        stateful_handler = AssessHandler(choice_map_sample)
        retval = forward(source_fn)(stateful_handler, *args)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper


############################
# Generate request handler #
############################


@dataclass
class GenerateHandler(StaticHandler):
    def __init__(self, key: PRNGKey, choice_map: ChoiceMap):
        super().__init__()
        self.key = key
        self.choice_map = choice_map
        self.weight: Weight = jnp.zeros(())
        self.key_counter = 1

    def fresh_key_and_increment(self):
        new_key = jax.random.fold_in(self.key, self.key_counter)
        self.key_counter += 1
        return new_key

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
    ) -> Constraint:
        return ChoiceMapConstraint(self.choice_map(addr))

    def handle_trace(
        self,
        addr: StaticAddress,
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        subconstraint = self.get_subconstraint(addr)
        sub_key = self.fresh_key_and_increment()
        (tr, w) = gen_fn.generate(sub_key, subconstraint, args)
        self.weight += w
        self.record(addr, tr)

        return tr.get_retval()


def generate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: tuple[Any, ...],
    ):
        stateful_handler = GenerateHandler(key, choice_map)
        retval = forward(source_fn)(stateful_handler, *args)
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


class UpdateHandler(StaticHandler):
    def __init__(
        self, key: PRNGKey, previous_trace: StaticTrace[Any], constraint: ChoiceMap
    ):
        super().__init__()
        self.key = key
        self.previous_trace = previous_trace
        self.constraint = constraint
        self.weight = jnp.zeros(())
        self.bwd_constraints: list[ChoiceMap] = []
        self.key_counter = 1

    def fresh_key_and_increment(self):
        new_key = jax.random.fold_in(self.key, self.key_counter)
        self.key_counter += 1
        return new_key

    def yield_state(self):
        return (
            self.weight,
            self.traces,
            self.bwd_constraints,
        )

    def get_subconstraint(self, addr: StaticAddress) -> ChoiceMap:
        return self.constraint(addr)

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
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        argdiffs: Argdiffs = args
        subtrace = self.get_subtrace(addr)
        constraint = self.get_subconstraint(addr)
        sub_key = self.fresh_key_and_increment()
        request = Update(constraint)
        (tr, w, retval_diff, bwd_request) = request.edit(
            sub_key,
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
        key: PRNGKey,
        previous_trace: StaticTrace[R],
        constraint: ChoiceMap,
        diffs: tuple[Any, ...],
    ):
        stateful_handler = UpdateHandler(key, previous_trace, constraint)
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


class StaticEditRequestHandler(StaticHandler):
    def __init__(
        self, key: PRNGKey, previous_trace: StaticTrace[Any], addressed: StaticDict
    ):
        super().__init__()
        self.key = key
        self.previous_trace = previous_trace
        self.addressed = addressed
        self.weight = jnp.zeros(())
        self.bwd_requests: list[EditRequest] = []
        self.key_counter = 1

    def fresh_key_and_increment(self):
        new_key = jax.random.fold_in(self.key, self.key_counter)
        self.key_counter += 1
        return new_key

    def yield_state(self):
        return (
            self.weight,
            self.traces,
            self.bwd_requests,
        )

    def get_subrequest(
        self, addr: StaticAddressComponent | StaticAddress
    ) -> EditRequest:
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
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        argdiffs: Argdiffs = args
        subtrace = self.get_subtrace(addr)
        subrequest = self.get_subrequest(addr)
        sub_key = self.fresh_key_and_increment()
        (tr, w, retval_diff, bwd_request) = subrequest.edit(
            sub_key,
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
        key: PRNGKey,
        previous_trace: StaticTrace[R],
        addressed: dict[StaticAddress, EditRequest],
        diffs: tuple[Any, ...],
    ):
        stateful_handler = StaticEditRequestHandler(
            key,
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


class RegenerateRequestHandler(StaticHandler):
    def __init__(
        self,
        key: PRNGKey,
        previous_trace: StaticTrace[Any],
        selection: Selection,
        edit_request: EditRequest,
    ):
        super().__init__()
        self.key = key
        self.previous_trace = previous_trace
        self.selection = selection
        self.edit_request = edit_request
        self.weight = jnp.zeros(())
        self.bwd_requests: list[EditRequest] = []
        self.key_counter = 1

    def fresh_key_and_increment(self):
        new_key = jax.random.fold_in(self.key, self.key_counter)
        self.key_counter += 1
        return new_key

    def yield_state(self):
        return (
            self.weight,
            self.traces,
            self.bwd_requests,
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
        gen_fn: GenerativeFunction[Any],
        args: tuple[Any, ...],
    ):
        argdiffs: Argdiffs = args
        subtrace = self.get_subtrace(addr)
        subselection = self.get_subselection(addr)
        sub_key = self.fresh_key_and_increment()
        subrequest = Regenerate(subselection)
        tr, w, retval_diff, bwd_request = subrequest.edit(sub_key, subtrace, argdiffs)
        self.bwd_requests.append(bwd_request)
        self.weight += w
        self.record(addr, tr)

        return retval_diff


def regenerate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(
        key: PRNGKey,
        previous_trace: StaticTrace[R],
        selection: Selection,
        edit_request: EditRequest,
        diffs: tuple[Any, ...],
    ):
        stateful_handler = RegenerateRequestHandler(
            key,
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


# Callee syntactic sugar handler.


def handler_trace_with_static(
    addr: StaticAddressComponent | StaticAddress,
    gen_fn: GenerativeFunction[Any],
    args: tuple[Any, ...],
):
    return trace(addr, gen_fn, args)


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

    def __get__(self, instance, _klass) -> "StaticGenerativeFunction[R]":
        """
        This method allows the @genjax.gen decorator to transform instance methods, turning them into `StaticGenerativeFunction[R]` calls.

        NOTE: if you assign an already-created `StaticGenerativeFunction` to a variable inside of a class, it will always receive the instance as its first method.
        """
        return self.partial_apply(instance) if instance else self

    # To get the type of return value, just invoke
    # the source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        return to_shape_fn(self.source, jnp.zeros)(*args)

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

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> StaticTrace[R]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (args, retval, traces) = simulate_transform(syntax_sugar_handled)(key, args)
        return StaticTrace(self, args, retval, traces)

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[StaticTrace[R], Weight]:
        assert isinstance(constraint, ChoiceMapConstraint), type(constraint)

        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )

        (
            weight,
            # Trace.
            (
                args,
                retval,
                traces,
            ),
        ) = generate_transform(syntax_sugar_handled)(key, constraint.choice_map, args)
        return StaticTrace(self, args, retval, traces), weight

    def project(
        self,
        key: PRNGKey,
        trace: Trace[Any],
        selection: Selection,
    ) -> Weight:
        assert isinstance(trace, StaticTrace)

        weight = jnp.array(0.0)
        for addr in trace.subtraces.keys():
            subprojection = selection(addr)
            subtrace = trace.get_subtrace(addr)
            weight += subtrace.project(key, subprojection)
        return weight

    def edit_update(
        self,
        key: PRNGKey,
        trace: StaticTrace[R],
        constraint: ChoiceMap,
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
                    traces,
                ),
                bwd_requests,
            ),
        ) = update_transform(syntax_sugar_handled)(key, trace, constraint, argdiffs)
        if not Diff.static_check_tree_diff(retval_diffs):
            retval_diffs = Diff.no_change(retval_diffs)

        def make_bwd_request(traces, subconstraints):
            addresses = traces.keys()
            chm = ChoiceMap.from_mapping(zip(addresses, subconstraints))
            return Update(chm)

        bwd_request = make_bwd_request(traces, bwd_requests)
        return (
            StaticTrace(
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
        key: PRNGKey,
        trace: StaticTrace[R],
        addressed: StaticDict,
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
                    traces,
                ),
                bwd_requests,
            ),
        ) = static_edit_request_transform(syntax_sugar_handled)(
            key, trace, addressed, argdiffs
        )

        def make_bwd_request(
            traces: dict[StaticAddress, Trace[R]],
            subrequests: list[EditRequest],
        ):
            return StaticRequest(dict(zip(traces.keys(), subrequests)))

        bwd_request = make_bwd_request(traces, bwd_requests)
        return (
            StaticTrace(
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
        key: PRNGKey,
        trace: StaticTrace[R],
        selection: Selection,
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
                    traces,
                ),
                bwd_requests,
            ),
        ) = regenerate_transform(syntax_sugar_handled)(
            key, trace, selection, edit_request, argdiffs
        )

        def make_bwd_request(
            traces: dict[StaticAddress, Trace[R]],
            subrequests: list[EditRequest],
        ):
            return StaticRequest(dict(zip(traces.keys(), subrequests)))

        bwd_request = make_bwd_request(traces, bwd_requests)
        return (
            StaticTrace(
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
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[StaticTrace[R], Weight, Retdiff[R], EditRequest]:
        assert isinstance(trace, StaticTrace)
        match edit_request:
            case Update(constraint):
                return self.edit_update(
                    key,
                    trace,
                    constraint,
                    argdiffs,
                )

            case StaticRequest(addressed):
                return self.edit_static_edit_request(
                    key,
                    trace,
                    addressed,
                    argdiffs,
                )
            case Regenerate(selection):
                return self.edit_regenerate(
                    key,
                    trace,
                    selection,
                    edit_request,
                    argdiffs,
                )
            case _:
                raise NotSupportedEditRequest(edit_request)

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

    @property
    def partial_args(self) -> tuple[Any, ...]:
        """
        Returns the partially applied arguments of the generative function.

        This method retrieves the dynamically applied arguments that were used to create
        this StaticGenerativeFunction instance through partial application.

        Returns:
            tuple[Any, ...]: A tuple containing the partially applied arguments.

        Note:
            This method is particularly useful when working with partially applied
            generative functions, allowing access to the pre-filled arguments.
        """
        return self.source.dyn_args

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
        all_args = self.source.dyn_args + args
        return gen(Closure[R](all_args, self.source.fn))


#############
# Decorator #
#############


def gen(f: Closure[R] | Callable[..., R]) -> StaticGenerativeFunction[R]:
    if isinstance(f, Closure):
        return StaticGenerativeFunction[R](f)
    else:
        closure = Closure[R]((), f)
        return gen(closure)


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
