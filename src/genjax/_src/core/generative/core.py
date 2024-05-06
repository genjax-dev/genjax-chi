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
from genjax._src.core.interpreters.staging import get_trace_data_shape, staged_and
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
    PRNGKey,
    Tuple,
    static_check_is_concrete,
)

register_exclusion(__file__)

Weight = Annotated[
    FloatArray,
    Is[lambda arr: arr.shape == ()],
]
Retdiff = Annotated[
    object,
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
class MaskUpdateSpec(UpdateSpec):
    flag: BoolArray
    spec: UpdateSpec

    @classmethod
    def maybe(cls, f: BoolArray, spec: UpdateSpec):
        match spec:
            case MaskUpdateSpec(flag, subspec):
                return MaskUpdateSpec(staged_and(f, flag), subspec)
            case _:
                static_bool_check = static_check_is_concrete(f) and isinstance(f, Bool)
                return (
                    spec
                    if static_bool_check and f
                    else EmptyUpdateSpec()
                    if static_bool_check
                    else MaskUpdateSpec(f, spec)
                )


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


@Pytree.dataclass
class MaskConstraint(Constraint):
    """
    A `MaskConstraint` encodes a possible constraint.

    Formally, `MaskConstraint(f: Bool, c: Constraint)` represents the constraint `Option((x $\\mapsto$ x, x))`,
    where the None case is represented by `EmptyConstraint`.
    """

    flag: BoolArray
    constraint: Constraint


@Pytree.dataclass
class SwitchConstraint(Constraint):
    """
    A `SwitchConstraint` encodes that one of a set of possible constraints is active _at runtime_, using a provided index.

    Formally, `SwitchConstraint(idx: IntArray, cs: List[Constraint])` represents the constraint (`x` $\\mapsto$ `xs[idx]`, `ys[idx]`).
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


@Pytree.dataclass
class MaskSample(Sample):
    flag: BoolArray
    sample: Sample

    def get_constraint(self) -> Constraint:
        return MaskConstraint(self.flag, self.sample.get_constraint())


#########
# Trace #
#########


class Trace(Pytree):
    """
    `Trace` is the abstract superclass for traces of generative functions.

    A trace is a data structure used to represent sampled executions of
    generative functions. Traces track metadata associated with
    log probabilities of choices, as well as other data associated with
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
    def get_score(self) -> FloatArray:
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
    ) -> Tuple["Trace", Weight, Retdiff, UpdateSpec]:
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, spec)

    def project(
        self,
        key: PRNGKey,
        spec: UpdateSpec,
    ) -> Weight:
        gen_fn = self.get_gen_fn()
        return gen_fn.project(key, self, spec)

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

    def get_trace_data_shape(self, *args) -> Any:
        return get_trace_data_shape(self, *args)

    def get_empty_trace(self, *args) -> Trace:
        data_shape = self.get_trace_data_shape(*args)
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
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        raise NotImplementedError

    @abstractmethod
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Weight, Any]:
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

    ###############################################
    # Convenience: postfix syntax for combinators #
    ###############################################

    def vmap(self, /, *, in_axes=0) -> "GenerativeFunction":
        from genjax import vmap_combinator

        return vmap_combinator(self, in_axes=in_axes)

    def repeat(self, /, *, num_repeats: Int) -> "GenerativeFunction":
        from genjax import repeat_combinator

        return repeat_combinator(self, num_repeats=num_repeats)

    def scan(self, /, *, max_length: Int) -> "GenerativeFunction":
        from genjax import scan_combinator

        return scan_combinator(self, max_length=max_length)

    def mask(self) -> "GenerativeFunction":
        from genjax import mask_combinator

        return mask_combinator(self)

    def or_else(self, gen_fn: "GenerativeFunction") -> "GenerativeFunction":
        from genjax import cond_combinator

        return cond_combinator(self, gen_fn)

    def addr_bij(
        self,
        address_bijection: dict,
    ) -> "GenerativeFunction":
        from genjax import address_bijection_combinator

        return address_bijection_combinator(self, address_bijection=address_bijection)

    def switch(self, *gen_fn: "GenerativeFunction") -> "GenerativeFunction":
        from genjax import switch_combinator

        return switch_combinator(self, *gen_fn)

    def mix(self, gen_fn: "GenerativeFunction") -> "GenerativeFunction":
        from genjax import mixture_combinator

        return mixture_combinator(self, gen_fn)

    def attach(self, **kwargs) -> "GenerativeFunction":
        from genjax.inference.smc import attach_combinator

        return attach_combinator(self, **kwargs)


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
class GenerativeFunctionClosure(Pytree):
    gen_fn: GenerativeFunction
    args: Tuple
    kwargs: Dict

    def get_gen_fn_with_kwargs(self):
        return self.gen_fn.handle_kwargs()

    def get_trace_data_shape(self) -> Any:
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.get_trace_data_shape(self.args, self.kwargs)
        else:
            return self.gen_fn.get_trace_data_shape(*self.args)

    def get_empty_trace(self) -> Trace:
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.get_empty_trace(self.args, self.kwargs)
        else:
            return self.gen_fn.get_empty_trace(*self.args)

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

    def __call__(self, key: PRNGKey) -> Any:
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key, (self.args, self.kwargs)
            ).get_retval()
        else:
            return self.gen_fn.simulate(key, self.args).get_retval()

    def __abstract_call__(self) -> Any:
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.__abstract_call__(*self.args, **self.kwargs)
        else:
            return self.gen_fn.__abstract_call__(*self.args)

    #############################################
    # Support the interface with reduced syntax #
    #############################################

    @GenerativeFunction.gfi_boundary
    def simulate(
        self,
        key: PRNGKey,
    ):
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key,
                (self.args, self.kwargs),
            )
        else:
            return self.gen_fn.simulate(key, self.args)

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ):
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.importance(
                key,
                constraint,
                (self.args, self.kwargs),
            )
        else:
            return self.gen_fn.importance(key, constraint, self.args)

    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        spec: UpdateSpec,
    ):
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.update(
                key,
                trace,
                spec,
                (self.args, self.kwargs),
            )
        else:
            return self.gen_fn.update(key, trace, spec, self.args)

    def assess(
        self,
        sample: Sample,
    ):
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.assess(
                sample,
                (self.args, self.kwargs),
            )
        else:
            return self.gen_fn.assess(sample, self.args)
