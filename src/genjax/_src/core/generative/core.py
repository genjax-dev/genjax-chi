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


from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import staged_and
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Annotated,
    Any,
    Bool,
    BoolArray,
    Callable,
    FloatArray,
    Generic,
    Int,
    IntArray,
    Is,
    List,
    ScalarFloat,
    Tuple,
    TypeVar,
    static_check_is_concrete,
)

register_exclusion(__file__)

#####################################
# Special generative function types #
#####################################

Weight = ScalarFloat

"""
A _weight_ is a density ratio which often occurs in the context of proper weighting for [`Target`][genjax.inference.Target] distributions, or in Gen's [`update`][genjax.core.GenerativeFunction.update] interface, whose mathematical content is described in [`update`][genjax.core.GenerativeFunction.update].

The type `Weight` does not enforce any meaningful mathematical invariants, but is used to denote the type of weights in GenJAX, to improve readability and parsing of interface specifications / expectations.
"""
Score = ScalarFloat
"""
A _score_ is a density ratio, described fully in [`simulate`][genjax.core.GenerativeFunction.simulate].

The type `Score` does not enforce any meaningful mathematical invariants, but is used to denote the type of scores in the GenJAX system, to improve readability and parsing of interface specifications.

Under type checking, the type `Score` enforces that the value must be a scalar floating point number.
"""

Arguments = Tuple
"""
`Arguments` is the type of argument values to generative functions. It is a type alias for `Tuple`, and is used to improve readability and parsing of interface specifications.
"""

Retval = Any
"""
`Retval` is the type of return values from the return value function of a generative function. It is a type alias for `Any`, and is used to improve readability and parsing of interface specifications.
"""

Argdiffs = Annotated[
    Tuple,
    Is[lambda v: Diff.static_check_tree_diff(v)],
]
"""
`Argdiffs` is the type of argument values with an attached `ChangeType` (c.f. [`update`][genjax.core.GenerativeFunction.update]).

When used under type checking, `Retdiff` assumes that the argument values are `Pytree` (either, defined via GenJAX's `Pytree` interface or registered with JAX's system). For each argument, it checks that _the leaves_ are `Diff` type with attached `ChangeType`.
"""


Retdiff = Annotated[
    Retval,
    Is[lambda v: Diff.static_check_tree_diff(v)],
]
"""
`Retdiff` is the type of return values with an attached `ChangeType` (c.f. [`update`][genjax.core.GenerativeFunction.update]).

When used under type checking, `Retdiff` assumes that the return value is a `Pytree` (either, defined via GenJAX's `Pytree` interface or registered with JAX's system). It checks that _the leaves_ are `Diff` type with attached `ChangeType`.
"""


#########################
# Update specifications #
#########################


class UpdateRequest(Pytree):
    """
    An `UpdateRequest` is a request to update a trace of a generative function. Generative functions respond to instances of subtypes of `UpdateRequest` by providing an [`update`][genjax.core.GenerativeFunction.update] implementation.

    Updating a trace is a common operation in inference processes, but naively mutating the trace will invalidate the mathematical invariants that Gen retains. `UpdateRequest` instances denote requests for _SMC moves_ in the framework of [SMCP3](https://proceedings.mlr.press/v206/lew23a.html), which preserve these invariants.
    """


@Pytree.dataclass
class EmptyUpdateRequest(UpdateRequest):
    pass


@Pytree.dataclass(match_args=True)
class MaskedUpdateRequest(UpdateRequest):
    flag: Bool | BoolArray
    request: UpdateRequest

    @classmethod
    def maybe_empty(cls, f: BoolArray, request: UpdateRequest):
        match request:
            case MaskedUpdateRequest(flag, subrequest):
                return MaskedUpdateRequest(staged_and(f, flag), subrequest)
            case _:
                static_bool_check = static_check_is_concrete(f) and isinstance(f, Bool)
                return (
                    request
                    if static_bool_check and f
                    else EmptyUpdateRequest()
                    if static_bool_check
                    else MaskedUpdateRequest(f, request)
                )


@Pytree.dataclass
class SumUpdateRequest(UpdateRequest):
    idx: Int | IntArray
    requests: List[UpdateRequest]


U = TypeVar("U", bound=UpdateRequest)

# NOTE: responding to this request is what old update does.
@Pytree.dataclass(match_args=True)
class IncrementalUpdateRequest(Generic[U], UpdateRequest):
    argdiffs: Argdiffs
    subrequest: U


C = TypeVar("C", bound="Constraint")


@Pytree.dataclass(match_args=True)
class ConstraintUpdateRequest(Generic[C], UpdateRequest):
    constraint: C

# NOTE: the importance interface is encapsulated by the new version of update.
# This request asks for that.
@Pytree.dataclass(match_args=True)
class ImportanceUpdateRequest(Generic[C], UpdateRequest):
    args: Arguments
    constraint: C


# NOTE: responding to this request is what Gen's project interface on traces does.
@Pytree.dataclass
class ProjectUpdateRequest(UpdateRequest):
    pass

# NOTE: responding to this request is what Gen's regenerate interface does.
@Pytree.dataclass
class RegenerateRequest(UpdateRequest):
    pass


class UpdateRequestBuilder(Pytree):
    @classmethod
    def empty(cls):
        return EmptyUpdateRequest()

    @classmethod
    def maybe(cls, flag: Bool | BoolArray, request: "UpdateRequest"):
        return MaskedUpdateRequest.maybe_empty(flag, request)

    @classmethod
    def g(cls, argdiffs: Argdiffs, subrequest: "UpdateRequest") -> "UpdateRequest":
        return IncrementalUpdateRequest(argdiffs, subrequest)


###############
# Constraints #
###############


class Constraint(Pytree):
    """
    `Constraint` is a type of [`UpdateRequest`][genjax.core.UpdateRequest] specified by a function from the [`Sample`][genjax.core.Sample] space of the generative function to a value space `Y`, and a target value `v` in `Y`. In other words, a [`Constraint`][genjax.core.Constraint] denotes the pair $(S \\mapsto Y, v \\in Y)$.

    Constraints represent a request to force a value to satisfy a predicate. Just like all [`UpdateRequest`][genjax.core.UpdateRequest] instances, the generative function must respond to the request to update a trace to satisfy the constraint by providing an [`update`][genjax.core.GenerativeFunction.update] implementation which implements an SMCP3 move that transforms the provided trace to satisfy the specification.

    Constraints can also be used to construct `ImportanceUpdateRequest` instances, which are used to implement the [`importance`][genjax.core.GenerativeFunction.importance] interface. This interface implements a restricted SMCP3 move, from the empty target, to the target induced by the constraint.
    """


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
    """A `Sample` is a value which can be sampled from generative functions. Samples can be scalar values, or map-like values ([`ChoiceMap`][genjax.core.ChoiceMap]). Different sample types can induce different interfaces: `ChoiceMap`, for instance, supports interfaces for accessing sub-maps and values."""


@Pytree.dataclass
class EmptySample(Sample):
    pass


@Pytree.dataclass(match_args=True)
class MaskedSample(Sample):
    flag: Bool | BoolArray
    sample: Sample
