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
    static_check_is_concrete,
    typecheck,
)

register_exclusion(__file__)

#####################################
# Special generative function types #
#####################################

Weight = Annotated[
    float | FloatArray,
    Is[lambda arr: arr.shape == ()],
]
Score = Annotated[
    float | FloatArray,
    Is[lambda arr: arr.shape == ()],
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


class UpdateSpec(Pytree):
    pass


@Pytree.dataclass
class EmptyUpdateSpec(UpdateSpec):
    pass


@Pytree.dataclass(match_args=True)
class ChangeTargetUpdateSpec(UpdateSpec):
    argdiffs: Tuple
    update_spec: UpdateSpec


@Pytree.dataclass(match_args=True)
class MaskedUpdateSpec(UpdateSpec):
    flag: Bool | BoolArray
    spec: UpdateSpec

    @classmethod
    def maybe(cls, f: BoolArray, spec: UpdateSpec):
        match spec:
            case MaskedUpdateSpec(flag, subspec):
                return MaskedUpdateSpec(staged_and(f, flag), subspec)
            case _:
                static_bool_check = static_check_is_concrete(f) and isinstance(f, Bool)
                return (
                    spec
                    if static_bool_check and f
                    else EmptyUpdateSpec()
                    if static_bool_check
                    else MaskedUpdateSpec(f, spec)
                )


@Pytree.dataclass
class SumUpdateSpec(UpdateSpec):
    idx: Int | IntArray
    specs: List[UpdateSpec]


@Pytree.dataclass
class RemoveSampleUpdateSpec(UpdateSpec):
    pass


###############
# Constraints #
###############


class Constraint(UpdateSpec):
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

    @abstractmethod
    def get_constraint(self) -> Constraint:
        pass


@Pytree.dataclass
class EmptySample(Sample):
    def get_constraint(self) -> Constraint:
        return EmptyConstraint()


@Pytree.dataclass(match_args=True)
class MaskedSample(Sample):
    flag: Bool | BoolArray
    sample: Sample

    def get_constraint(self) -> Constraint:
        return MaskedConstraint(self.flag, self.sample.get_constraint())


#########
# Trace #
#########


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
    def get_sample(self) -> Sample:
        """Return a `Sample`, a representation of the sample from the measure denoted by the generative function.

        Examples:
        """

    @abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        """Returns the generative function whose invocation created the `Trace`.

        Examples:
        """
        raise NotImplementedError

    def update(
        self,
        key: PRNGKey,
        spec: UpdateSpec,
        argdiffs: Optional[Tuple | Argdiffs] = None,
    ) -> Tuple["Trace", Weight, Retdiff, UpdateSpec]:
        gen_fn = self.get_gen_fn()
        if argdiffs:
            check = Diff.static_check_tree_diff(argdiffs)
            argdiffs = argdiffs if check else Diff.tree_diff_unknown_change(argdiffs)
            return gen_fn.update(key, self, spec, argdiffs)
        else:
            old_args = self.get_args()
            argdiffs = Diff.tree_diff_no_change(old_args)
            return gen_fn.update(key, self, spec, argdiffs)

    def project(
        self,
        key: PRNGKey,
        spec: UpdateSpec,
    ) -> Weight:
        gen_fn = self.get_gen_fn()
        return gen_fn.project(key, self, spec)

    ##########################
    # UpdateCompiler interface #
    ##########################

    def create_update_spec(self, addr, v) -> UpdateSpec:
        raise NotImplementedError

    @Pytree.dataclass
    class UpdateCompiler(Pytree):
        trace: "Trace"
        addr: Any
        updates: List[UpdateSpec]

        def __getitem__(self, addr) -> "Trace.UpdateCompiler":
            return Trace.UpdateCompiler(
                self.trace,
                addr,
                self.updates,
            )

        def set(self, v) -> "Trace.UpdateCompiler":
            new_spec = self.trace.create_update_spec(self.addr, v)
            return Trace.UpdateCompiler(self.trace, [], [*self.updates, new_spec])

        @property
        def at(self) -> "Trace.UpdateCompiler":
            return self

        def update(self, key) -> Tuple["Trace", Weight, List[UpdateSpec]]:
            trace = self.trace
            w = 0.0
            bwd_specs = []
            for update in self.updates:
                trace, inc_w, _, bwd_spec = trace.update(key, update)
                w += inc_w
                bwd_specs.append(bwd_spec)

            return trace, w, list(reversed(bwd_specs))

    @property
    def at(self) -> UpdateCompiler:
        return Trace.UpdateCompiler(self, [], [])

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

    def summary(self):
        return TraceSummary(
            self.get_gen_fn(),
            self.get_sample(),
            self.get_score(),
            self.get_retval(),
        )


@Pytree.dataclass
class TraceSummary(Pytree):
    gen_fn: "GenerativeFunction"
    sample: Sample
    score: FloatArray
    retval: Any

    def treescope_color(self):
        return self.gen_fn.treescope_color()


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
            from jax.random import PRNGKey

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
            sub_keys = jax.random.split(key, 10)
            sim = model.repeat(num_repeats=10).simulate
            tr = jax.jit(jax.vmap(sim, in_axes=(0, None)))(sub_keys, ())
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
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        raise NotImplementedError

    @abstractmethod
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        raise NotImplementedError

    # TODO: check the math.
    def project(
        self,
        key: PRNGKey,
        trace: Trace,
        spec: UpdateSpec,
    ) -> Weight:
        args = trace.get_args()
        argdiffs = Diff.tree_diff_no_change(args)
        _, w, _, _ = self.update(key, trace, spec, argdiffs)
        return -w

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
            else scan_combinator(self, max_length=max_length)(*args)
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
        (args, kwargs) = args
        return self.wrapped.simulate(key, args)

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ):
        (args, kwargs) = args
        return self.wrapped.importance(key, constraint, args)

    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: Constraint,
        argdiffs: Tuple,
    ):
        (argdiffs, kwargdiffs) = argdiffs
        return self.wrapped.update(key, trace, update_spec, argdiffs)


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
                key, (full_args, self.kwargs)
            ).get_retval()
        else:
            return self.gen_fn.simulate(key, self.args).get_retval()

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
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.importance(
                key,
                constraint,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.importance(key, constraint, full_args)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        spec: UpdateSpec,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        full_argdiffs = (*self.args, *argdiffs)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.update(
                key,
                trace,
                spec,
                (full_argdiffs, self.kwargs),
            )
        else:
            return self.gen_fn.update(key, trace, spec, full_argdiffs)

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
