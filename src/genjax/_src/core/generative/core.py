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

from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import Flag
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Annotated,
    Any,
    FloatArray,
    Generic,
    IntArray,
    Is,
    TypeVar,
)

# Generative Function type variables
R = TypeVar("R")
"""
Generic denoting the return type of a generative function.
"""
S = TypeVar("S")


#####################################
# Special generative function types #
#####################################

Weight = FloatArray
"""
A _weight_ is a density ratio which often occurs in the context of proper weighting for [`Target`][genjax.inference.Target] distributions, or in Gen's [`edit`][genjax.core.GenerativeFunction.edit] interface, whose mathematical content is described in [`edit`][genjax.core.GenerativeFunction.edit].

The type `Weight` does not enforce any meaningful mathematical invariants, but is used to denote the type of weights in GenJAX, to improve readability and parsing of interface specifications / expectations.
"""
Score = FloatArray
"""
A _score_ is a density ratio, described fully in [`simulate`][genjax.core.GenerativeFunction.simulate].

The type `Score` does not enforce any meaningful mathematical invariants, but is used to denote the type of scores in the GenJAX system, to improve readability and parsing of interface specifications.
"""

Arguments = tuple
"""
`Arguments` is the type of argument values to generative functions. It is a type alias for `Tuple`, and is used to improve readability and parsing of interface specifications.
"""

Argdiffs = Annotated[
    tuple[Any, ...],
    Is[Diff.static_check_tree_diff],
]
"""
`Argdiffs` is the type of argument values with an attached `ChangeType` (c.f. [`edit`][genjax.core.GenerativeFunction.edit]).

When used under type checking, `Retdiff` assumes that the argument values are `Pytree` (either, defined via GenJAX's `Pytree` interface or registered with JAX's system). For each argument, it checks that _the leaves_ are `Diff` type with attached `ChangeType`.
"""


Retdiff = Annotated[
    R,
    Is[Diff.static_check_tree_diff],
]


"""
`Retdiff` is the type of return values with an attached `ChangeType` (c.f. [`edit`][genjax.core.GenerativeFunction.edit]).

When used under type checking, `Retdiff` assumes that the return value is a `Pytree` (either, defined via GenJAX's `Pytree` interface or registered with JAX's system). It checks that _the leaves_ are `Diff` type with attached `ChangeType`.
"""

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
    flag: Flag
    sample: Sample


###############
# Constraints #
###############


class Constraint(Pytree):
    """
    Constraints represent a request to force a value to satisfy a predicate.
    """


@Pytree.dataclass
class EmptyConstraint(Constraint):
    """
    An `EmptyConstraint` encodes the lack of a constraint.

    Formally, `EmptyConstraint(x)` represents the constraint `(x $\\mapsto$ (), ())`.
    """

    pass


@Pytree.dataclass(match_args=True)
class MaskedConstraint(Constraint):
    """
    A `MaskedConstraint` encodes a possible constraint.

    Formally, `MaskedConstraint(f: Bool, c: Constraint)` represents the constraint `Option((x $\\mapsto$ x, x))`,
    where the None case is represented by `EmptyConstraint`.
    """

    idx: IntArray
    constraint: list[Constraint]


###############
# Projections #
###############


class Projection(Generic[S], Pytree):
    @abstractmethod
    def filter(self, sample: S) -> S:
        raise NotImplementedError

    @abstractmethod
    def complement(self) -> "Projection[S]":
        raise NotImplementedError


#########################
# Update specifications #
#########################


class EditRequest(Pytree):
    """
    An `EditRequest` is a request to edit a trace of a generative function. Generative functions respond to instances of subtypes of `EditRequest` by providing an [`edit`][genjax.core.GenerativeFunction.edit] implementation.

    Updating a trace is a common operation in inference processes, but naively mutating the trace will invalidate the mathematical invariants that Gen retains. `EditRequest` instances denote requests for _SMC moves_ in the framework of [SMCP3](https://proceedings.mlr.press/v206/lew23a.html), which preserve these invariants.
    """


@Pytree.dataclass
class EmptyRequest(EditRequest):
    pass


@Pytree.dataclass(match_args=True)
class IncrementalGenericRequest(EditRequest):
    constraint: Constraint
