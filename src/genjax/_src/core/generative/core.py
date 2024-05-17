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

from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from penzai.core import formatting_util

from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import get_trace_shape, staged_and
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import gfi_boundary, register_exclusion
from genjax._src.core.typing import (
    Annotated,
    Any,
    Bool,
    BoolArray,
    Callable,
    Dict,
    FloatArray,
    Int,
    IntArray,
    Is,
    List,
    Optional,
    PRNGKey,
    Tuple,
    TypeVar,
    static_check_is_concrete,
    typecheck,
)

register_exclusion(__file__)

#####################################
# Special generative function types #
#####################################

Weight = Annotated[
    float | FloatArray,
    Is[lambda arr: jnp.array(arr, copy=False).shape == ()],
]
Score = Annotated[
    float | FloatArray,
    Is[lambda arr: jnp.array(arr, copy=False).shape == ()],
]
Retdiff = Annotated[
    object,
    Is[lambda v: Diff.static_check_tree_diff(v)],
]
Argdiffs = Annotated[
    Tuple,
    Is[lambda v: Diff.static_check_tree_diff(v)],
]

#########################
# Update specifications #
#########################


class UpdateProblem(Pytree):
    @classmethod
    def empty(cls):
        return EmptyProblem()

    @classmethod
    def maybe(cls, flag: Bool | BoolArray, problem: "UpdateProblem"):
        return MaskedProblem.maybe_empty(flag, problem)


@Pytree.dataclass
class EmptyProblem(UpdateProblem):
    pass


@Pytree.dataclass(match_args=True)
class MaskedProblem(UpdateProblem):
    flag: Bool | BoolArray
    problem: UpdateProblem

    @classmethod
    def maybe_empty(cls, f: BoolArray, problem: UpdateProblem):
        match problem:
            case MaskedProblem(flag, subproblem):
                return MaskedProblem(staged_and(f, flag), subproblem)
            case _:
                static_bool_check = static_check_is_concrete(f) and isinstance(f, Bool)
                return (
                    problem
                    if static_bool_check and f
                    else EmptyProblem()
                    if static_bool_check
                    else MaskedProblem(f, problem)
                )


@Pytree.dataclass
class SumProblem(UpdateProblem):
    idx: Int | IntArray
    problems: List[UpdateProblem]


@Pytree.dataclass(match_args=True)
class ImportanceProblem(UpdateProblem):
    constraint: "Constraint"


@Pytree.dataclass
class ProjectProblem(UpdateProblem):
    pass


###############
# Constraints #
###############


class Constraint(UpdateProblem):
    pass


@Pytree.dataclass
class EmptyConstraint(Constraint):
    """
    An `EmptyConstraint` encodes the lack of a constraint.

    Formally, `EmptyConstraint(x)` represents the constraint `(x $\\mapsto$ (), ())`.
    """

    pass


@Pytree.dataclass
class EqualityConstraint(Constraint):
    """
    An `EqualityConstraint` encodes the constraint that the value output by a
    distribution is equal to a provided value.

    Formally, `EqualityConstraint(x)` represents the constraint `(x $\\mapsto$ x, x)`.
    """

    x: Any


@Pytree.dataclass(match_args=True)
class MaskedConstraint(Constraint):
    """
    A `MaskedConstraint` encodes a possible constraint.

    Formally, `MaskedConstraint(f: Bool, c: Constraint)` represents the constraint `Option((x $\\mapsto$ x, x))`,
    where the None case is represented by `EmptyConstraint`.
    """

    flag: Bool | BoolArray
    constraint: Constraint


@Pytree.dataclass
class SumConstraint(Constraint):
    """
    A `SumConstraint` encodes that one of a set of possible constraints is active _at runtime_, using a provided index.

    Formally, `SumConstraint(idx: IntArray, cs: List[Constraint])` represents the constraint (`x` $\\mapsto$ `xs[idx]`, `ys[idx]`).
    """

    idx: IntArray
    constraint: List[Constraint]


@Pytree.dataclass
class IntervalConstraint(Constraint):
    """
    An IntervalConstraint encodes the constraint that the value output by a
    distribution on the reals lies within a given interval.

    Formally, `IntervalConstraint(a, b)` represents the constraint (`x` $\\mapsto$ `a` $\\leq$ `x` $\\leq$ `b`, `True`).
    """

    a: FloatArray
    b: FloatArray


@Pytree.dataclass
class BijectiveConstraint(Constraint):
    """
    A `BijectiveConstraint` encodes the constraint that the value output by a distribution
    must, under a bijective transformation, be equal to the value provided to the constraint.

    Formally, `BijectiveConstraint(bwd, v)` represents the constraint `(x $\\mapsto$ inverse(bwd)(x), v)`.
    """

    bwd: Callable[[Any], "Sample"]
    v: Any


###########
# Samples #
###########


class Sample(Pytree):
    """`Sample` is the abstract base class of the type of values which can be sampled from generative functions."""


@Pytree.dataclass
class EmptySample(Sample):
    pass


@Pytree.dataclass(match_args=True)
class MaskedSample(Sample):
    flag: Bool | BoolArray
    sample: Sample


#########
# Trace #
#########

T = TypeVar("T", bound=Sample)


class Trace(Pytree):
    """
    `Trace` is the abstract superclass for traces of generative functions.

    A trace is a data structure used to represent sampled executions of
    generative functions. Traces track metadata associated with the probabilities
    of choices, as well as other data associated with
    the invocation of a generative function, including the arguments it
    was invoked with, its return value, and the identity of the generative function itself.
    """

    @abstractmethod
    def get_args(self) -> Tuple:
        """Returns the arguments for the generative function invocation which
        created the `Trace`.
        """

    @abstractmethod
    def get_retval(self) -> Any:
        """Returns the return value from the generative function invocation which
        created the `Trace`.

        Examples:
            Here's an example using `genjax.normal` (a distribution). For distributions, the return value is the same as the (only) value in the returned choice map.

        """

    @abstractmethod
    def get_score(self) -> Score:
        """Return the score of the `Trace`.

        Examples:
        """

    @abstractmethod
    def get_sample(self) -> T:
        """Return a `Sample`, a representation of the sample from the measure denoted by the generative function.

        Examples:
        """

    # TODO: deprecated.
    def get_choices(self) -> Sample:
        return self.get_sample()

    @abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        """Returns the generative function whose invocation created the `Trace`.

        Examples:
        """
        raise NotImplementedError

    def update(
        self,
        key: PRNGKey,
        problem: UpdateProblem,
        argdiffs: Optional[Tuple | Argdiffs] = None,
    ) -> Tuple["Trace", Weight, Retdiff, UpdateProblem]:
        gen_fn = self.get_gen_fn()
        if argdiffs:
            check = Diff.static_check_tree_diff(argdiffs)
            argdiffs = argdiffs if check else Diff.tree_diff_unknown_change(argdiffs)
            return gen_fn.update(key, self, problem, argdiffs)
        else:
            old_args = self.get_args()
            argdiffs = Diff.tree_diff_no_change(old_args)
            return gen_fn.update(key, self, problem, argdiffs)

    @typecheck
    def project(
        self,
        key: PRNGKey,
        problem: ProjectProblem,
    ) -> Weight:
        gen_fn = self.get_gen_fn()
        _, w, _, _ = gen_fn.update(key, self, problem, Diff.no_change(self.get_args()))
        return -w

    ###################
    # Pretty printing #
    ###################

    def treescope_color(self):
        return self.get_gen_fn().treescope_color()

    ###################
    # Batch semantics #
    ###################

    @property
    def batch_shape(self):
        return len(self.get_score())


@Pytree.dataclass
class EmptyTraceArg(Pytree):
    pass


@Pytree.dataclass
class EmptyTraceRetval(Pytree):
    pass


@Pytree.dataclass
class EmptyTrace(Trace):
    gen_fn: "GenerativeFunction"

    def get_args(self) -> Tuple:
        return (EmptyTraceArg(),)

    def get_retval(self) -> Any:
        return EmptyTraceRetval()

    def get_score(self) -> Score:
        return 0.0

    def get_sample(self) -> Sample:
        return EmptySample()

    def get_gen_fn(self) -> "GenerativeFunction":
        return self.gen_fn


#######################
# Generative function #
#######################


class GenerativeFunction(Pytree):
    def __call__(self, *args, **kwargs) -> "GenerativeFunctionClosure":
        return GenerativeFunctionClosure(self, args, kwargs)

    def __abstract_call__(self, *args) -> Any:
        """Used to support JAX tracing, although this default implementation involves no
        JAX operations (it takes a fixed-key sample from the return value).

        Generative functions may customize this to improve compilation time.
        """
        return self.simulate(jax.random.PRNGKey(0), args).get_retval()

    def handle_kwargs(self) -> "GenerativeFunction":
        return IgnoreKwargs(self)

    def get_trace_shape(self, *args) -> Any:
        return get_trace_shape(self, args)

    def get_empty_trace(self, *args) -> Trace:
        data_shape = self.get_trace_shape(*args)
        return jtu.tree_map(lambda v: jnp.zeros(v.shape, dtype=v.dtype), data_shape)

    @classmethod
    def gfi_boundary(cls, c: Callable) -> Callable:
        return gfi_boundary(c)

    @abstractmethod
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        """
        Execute the generative function, which may include sampling random choices, and return a [`Trace`](core.md#genjax.core.Trace).

        Examples:
            ```python exec="yes" html="true" source="material-block" session="core"
            import genjax
            from jax import vmap, jit
            from jax.random import PRNGKey
            from jax.random import split

            @genjax.gen
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                return x

            key = PRNGKey(0)
            tr = model.simulate(key, ())
            print(tr.render_html())
            ```

            Another example, using the same model, composed into [`genjax.repeat_combinator`](generative_functions.md) - which creates a new generative function, which has the same interface:
            ```python exec="yes" html="true" source="material-block" session="core"
            @genjax.gen
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                return x

            key = PRNGKey(0)
            tr = model.repeat(num_repeats=10).simulate(key, ())
            print(tr.render_html())
            ```

            (**Fun, flirty, fast ... parallel?**) Feel free to use `jax.jit` and `jax.vmap`!
            ```python exec="yes" html="true" source="material-block" session="core"
            key = PRNGKey(0)
            sub_keys = split(key, 10)
            sim = model.repeat(num_repeats=10).simulate
            tr = jit(vmap(sim, in_axes=(0, None)))(sub_keys, ())
            print(tr.render_html())
            ```


        The trace returned by `simulate` has the arguments of the invocation ([`Trace.get_args`](core.md#genjax.core.Trace.get_args)), the return value of the generative function ([`Trace.get_retval`](core.md#genjax.core.Trace.get_retval)), the identity of the generative function which produced the trace ([`Trace.get_gen_fn`](core.md#genjax.core.Trace.get_gen_fn)), the sample of traced random choices produced during the invocation ([`Trace.get_sample`](core.md#genjax.core.Trace.get_sample)) and _the score_ of the sample ([`Trace.get_score`](core.md#genjax.core.Trace.get_score)).

        The score must satisfy a particular mathematical specification.

        Denote the sample by $t$ and the arguments by $a$: when the generative function contains no _untraced randomness_, the score (in logspace) is given by:

        $$
        s := \\log p(t; a)
        $$

        (**With untraced randomness**) Gen allows for the possibility of sources of randomness _which are not traced_. In GenJAX, this might look something like:
        ```python
        # notice how the key is explicit
        @genjax.gen
        def model_with_untraced_randomness(key: PRNGKey):
            x = genjax.normal(0.0, 1.0) "x"
            v = some_random_process(key, x)
            y = genjax.normal(v, 1.0) @ "y"
        ```

        In that case, the score is given by:

        $$
        s := \\log p(r, t; a) - \\log q(r; a)
        $$
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        key: PRNGKey,
        trace: EmptyTrace | Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        raise NotImplementedError

    @abstractmethod
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        raise NotImplementedError

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight]:
        importance_problem = ImportanceProblem(constraint)
        tr, w, _, _ = self.update(
            key, EmptyTrace(self), importance_problem, Diff.unknown_change(args)
        )
        return tr, w

    # NOTE: Supports pretty printing in penzai.
    def treescope_color(self):
        type_string = str(type(self))
        return formatting_util.color_from_string(type_string)

    ######################################################
    # Convenience: postfix syntax for combinators / DSLs #
    ######################################################

    ###############
    # Combinators #
    ###############

    def vmap(
        self,
        *args,
        in_axes=0,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import vmap_combinator

        return (
            vmap_combinator(self, in_axes=in_axes)(*args)
            if args
            else vmap_combinator(self, in_axes=in_axes)
        )

    def repeat(
        self,
        *args,
        num_repeats: Int,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import repeat_combinator

        return (
            repeat_combinator(self, num_repeats=num_repeats)(*args)
            if args
            else repeat_combinator(self, num_repeats=num_repeats)
        )

    def scan(
        self,
        *args,
        max_length: Int,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import scan_combinator

        return (
            scan_combinator(self, max_length=max_length)(*args)
            if args
            else scan_combinator(self, max_length=max_length)
        )

    def mask(
        self,
        *args,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import mask_combinator

        return mask_combinator(self)(*args) if args else mask_combinator(self)

    def or_else(
        self,
        gen_fn: "GenerativeFunction",
        *args,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import cond_combinator

        return (
            cond_combinator(self, gen_fn)(*args)
            if args
            else cond_combinator(self, gen_fn)
        )

    def addr_bij(
        self,
        address_bijection: dict,
        *args,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import address_bijection_combinator

        return (
            address_bijection_combinator(self, address_bijection=address_bijection)(
                *args
            )
            if args
            else address_bijection_combinator(self, address_bijection=address_bijection)
        )

    def switch(
        self,
        branches: List["GenerativeFunction"],
        *args,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import switch_combinator

        return (
            switch_combinator(self, *branches)(*args)
            if args
            else switch_combinator(self, *branches)
        )

    def mix(
        self,
        gen_fn: "GenerativeFunction",
        *args,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import mixture_combinator

        return (
            mixture_combinator(self, gen_fn)(*args)
            if args
            else mixture_combinator(self, gen_fn)
        )

    def attach(
        self,
        *args,
        **kwargs,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax.inference.smc import attach_combinator

        return (
            attach_combinator(self, **kwargs)(*args)
            if args
            else attach_combinator(self, **kwargs)
        )

    #####################
    # GenSP / inference #
    #####################

    def marginal(
        self,
        *args,
        select_or_addr: Optional[Any] = None,
        algorithm: Optional[Any] = None,
    ) -> "GenerativeFunction | GenerativeFunctionClosure":
        from genjax import marginal

        return (
            marginal(self, select_or_addr=select_or_addr, algorithm=algorithm)(*args)
            if args
            else marginal(self, select_or_addr=select_or_addr, algorithm=algorithm)
        )

    def target(
        self,
        /,
        *,
        constraint: Constraint,
        args: Tuple,
    ):
        from genjax import Target

        return Target(
            self,
            args,
            constraint,
        )


# NOTE: Setup a global handler stack for the `trace` callee sugar.
# C.f. above.
# This stack will not interact with JAX tracers at all
# so it's safe, and will be resolved at JAX tracing time.
GLOBAL_TRACE_OP_HANDLER_STACK: List[Callable] = []


def handle_off_trace_stack(addr, gen_fn: GenerativeFunction, args):
    if GLOBAL_TRACE_OP_HANDLER_STACK:
        handler = GLOBAL_TRACE_OP_HANDLER_STACK[-1]
        return handler(addr, gen_fn, args)
    else:
        raise Exception(
            "Attempting to invoke trace outside of a tracing context.\nIf you want to invoke the generative function closure, and recieve a return value,\ninvoke it with a key."
        )


def push_trace_overload_stack(handler, fn):
    def wrapped(*args):
        GLOBAL_TRACE_OP_HANDLER_STACK.append(handler)
        ret = fn(*args)
        GLOBAL_TRACE_OP_HANDLER_STACK.pop()
        return ret

    return wrapped


@Pytree.dataclass
class IgnoreKwargs(GenerativeFunction):
    wrapped: "GenerativeFunction"

    def handle_kwargs(self) -> "GenerativeFunction":
        raise NotImplementedError

    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ):
        (args, _kwargs) = args
        return self.wrapped.simulate(key, args)

    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: Constraint,
        argdiffs: Tuple,
    ):
        (argdiffs, _kwargdiffs) = argdiffs
        return self.wrapped.update(key, trace, update_problem, argdiffs)


@Pytree.dataclass
class GenerativeFunctionClosure(GenerativeFunction):
    gen_fn: GenerativeFunction
    args: Tuple
    kwargs: Dict

    def get_gen_fn_with_kwargs(self):
        return self.gen_fn.handle_kwargs()

    # NOTE: Supports callee syntax, and the ability to overload it in callers.
    def __matmul__(self, addr):
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return handle_off_trace_stack(
                addr,
                maybe_kwarged_gen_fn,
                (self.args, self.kwargs),
            )
        else:
            return handle_off_trace_stack(
                addr,
                self.gen_fn,
                self.args,
            )

    def __call__(self, key: PRNGKey, *args) -> Any:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key, (*full_args, self.kwargs)
            ).get_retval()
        else:
            return self.gen_fn.simulate(key, full_args).get_retval()

    def __abstract_call__(self, *args) -> Any:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.__abstract_call__(*full_args, **self.kwargs)
        else:
            return self.gen_fn.__abstract_call__(*full_args)

    #############################################
    # Support the interface with reduced syntax #
    #############################################

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.simulate(key, full_args)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        full_argdiffs = (*self.args, *argdiffs)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.update(
                key,
                trace,
                problem,
                (full_argdiffs, self.kwargs),
            )
        else:
            return self.gen_fn.update(key, trace, problem, full_argdiffs)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.assess(
                sample,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.assess(sample, full_args)
