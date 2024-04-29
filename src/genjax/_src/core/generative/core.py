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

from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    BoolArray,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
)

Weight = FloatArray
ReturnValue = Any
Retdiff = Any


###########
# Samples #
###########


class Sample(Pytree):
    """`Sample` is the abstract base class of the type of values which can be sampled from generative functions."""

    pass


#########################
# Update specifications #
#########################


class UpdateSpec(Pytree):
    pass


class ProjectSpec(UpdateSpec):
    """
    A `ProjectSpec` is the reverse move for `GenerativeFunction.importance`. Denotes that a call to `Trace.update` should return a weight computation of the following form:
    """

    pass


class RegenerateSpec(UpdateSpec):
    pass


class ChangeTargetConstraintSpec(UpdateSpec):
    argdiffs: Tuple
    new_constraints: "Constraint"


###############
# Constraints #
###############


class Constraint(UpdateSpec):
    pass


@Pytree.dataclass
class EmptyConstraint(Constraint):
    """
    An `EmptyConstraint` encodes the lack of a constraint.

    Formally, `EmptyConstraint(x)` represents the constraint `(x \mapsto (), ())`.
    """

    pass


@Pytree.dataclass
class EqualityConstraint(Constraint):
    """
    An `EqualityConstraint` encodes the constraint that the value output by a
    distribution is equal to a provided value.

    Formally, `EqualityConstraint(x)` represents the constraint `(x \mapsto x, x)`.
    """

    x: Any


@Pytree.dataclass
class MaybeConstraint(Constraint):
    """
    A `MaybeConstraint` encodes a possible constraint.

    Formally, `MaybeConstraint(f: Bool, c: Constraint)` represents the constraint `Option((x \mapsto x, x))`,
    where the None case is represented by `EmptyConstraint`.
    """

    flag: BoolArray
    constraint: Constraint


@Pytree.dataclass
class IntervalConstraint(Constraint):
    """
    An IntervalConstraint encodes the constraint that the value output by a
    distribution on the reals lies within a given interval.

    Formally, `IntervalConstraint(a, b)` represents the constraint `(x \mapsto a <= x <= b, True)`.
    """

    a: FloatArray
    b: FloatArray


@Pytree.dataclass
class BijectiveConstraint(Constraint):
    """
    A `BijectiveConstraint` encodes the constraint that the value output by a distribution
    must, under a bijective transformation, be equal to the value provided to the constraint.

    Formally, `BijectiveConstraint(bwd, v)` represents the constraint `(x \mapsto inverse(bwd)(x), v)`.
    """

    bwd: Callable[[Any], Sample]
    v: Any


#########
# Trace #
#########


class Trace(Pytree):
    """> Abstract base class for traces of generative functions.

    A `Trace` is a data structure used to represent sampled executions
    of generative functions.

    Traces track metadata associated with log probabilities of choices,
    as well as other data associated with the invocation of a generative
    function, including the arguments it was invoked with, its return
    value, and the identity of the generative function itself.
    """

    @abstractmethod
    def get_retval(self) -> Any:
        """Returns the return value from the generative function invocation which
        created the `Trace`.

        Examples:
            Here's an example using `genjax.normal` (a distribution). For distributions, the return value is the same as the (only) value in the returned choice map.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            retval = tr.get_retval()
            sample: ChoiceMap = tr.get_sample()
            v = sample.get_value()
            print(console.render((retval, v)))
            ```
        """

    @abstractmethod
    def get_score(self) -> FloatArray:
        """Return the score of the `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli

            console = genjax.console()


            @genjax.static_gen_fn
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x


            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            score = tr.get_score()
            x_score = bernoulli.logpdf(tr["x"], 0.3)
            y_score = bernoulli.logpdf(tr["y"], 0.3)
            print(console.render((score, x_score + y_score)))
            ```
        """

    @abstractmethod
    def get_sample(self) -> Sample:
        """Return a `Sample`, a representation of the sample from the measure denoted by the generative function.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli

            console = genjax.console()


            @genjax.static_gen_fn
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x


            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            choice = tr.get_sample()
            print(console.render(choice))
            ```
        """

    @abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        """Returns the generative function whose invocation created the `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            gen_fn = tr.get_gen_fn()
            print(console.render(gen_fn))
            ```
        """

    def update(
        self,
        key: PRNGKey,
        spec: UpdateSpec,
    ) -> Tuple["Trace", Weight, UpdateSpec]:
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


####################
# Posterior target #
####################


@Pytree.dataclass
class Target(Pytree):
    """
    A `Target` represents a constrained distribution, which may or may not be normalized.

    A `Target` is created by pairing a generative function and its arguments with a `Constraint` object.
    """

    gen_fn: "GenerativeFunction"
    constraints: Constraint

    @classmethod
    def n(cls, gen_fn) -> "Target":
        return Target(gen_fn, EmptyConstraint())


###########################
# Properly weighted trace #
###########################


@Pytree.dataclass
class ProperlyWeightedTrace(Pytree):
    """
    A `ProperlyWeightedTrace` is a triple containing an importance weight `w`, a `Trace`, and a `Target`.
    """

    w: FloatArray
    trace: Trace
    target: Target


#######################
# Generative function #
#######################


class GenerativeFunction(Pytree):
    def __abstract_call__(self) -> Any:
        """Used to support JAX tracing, although this default implementation involves no
        JAX operations (it takes a fixed-key sample from the return value).

        Generative functions may customize this to improve compilation time.
        """
        return self.simulate(jax.random.PRNGKey(0)).get_retval()

    @abstractmethod
    def simulate(
        self,
        key: PRNGKey,
    ) -> Trace:
        raise NotImplementedError

    @abstractmethod
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        raise NotImplementedError

    def project(
        self,
        key: PRNGKey,
        trace: Trace,
        spec: UpdateSpec,
    ) -> Weight:
        _, w, _, _ = self.update(key, trace, spec)
        return w
