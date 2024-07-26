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
"""This module contains the `Distribution` abstract base class."""

from abc import abstractmethod

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Argdiffs,
    Arguments,
    Assessable,
    ChoiceMapCoercable,
    ChoiceMapConstraint,
    ChoiceMapSample,
    Constraint,
    EmptyChm,
    EmptyConstraint,
    EmptySample,
    EmptyTrace,
    EmptyUpdateRequest,
    EqualityConstraint,
    GeneralRegenerateRequest,
    GeneralUpdateRequest,
    GenerativeFunction,
    IdentityProjection,
    ImportanceRequest,
    MaskedConstraint,
    MaskedSample,
    Projection,
    ProjectRequest,
    Retdiff,
    Retval,
    Sample,
    Score,
    SelectionProjection,
    Simulateable,
    Trace,
    UpdateRequest,
    ValChm,
    ValueSample,
    Weight,
)
from genjax._src.core.generative.core import (
    EmptyProjection,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    PRNGKey,
    TypeVar,
    tuple,
)

A = TypeVar("A", bound=Arguments)
R = TypeVar("R", bound=Retval)

#####
# DistributionTrace
#####


@Pytree.dataclass
class DistributionTrace(
    Generic[A, R],
    ChoiceMapCoercable,
    Trace["Distribution[A, R]", A, "ValueSample[R]", R],
):
    gen_fn: "Distribution[A, R]"
    arguments: A
    value: R
    score: Score

    def get_args(self) -> A:
        return self.arguments

    def get_retval(self) -> R:
        return self.value

    def get_gen_fn(self) -> "Distribution[A, R]":
        return self.gen_fn

    def get_score(self) -> Score:
        return self.score

    def get_sample(self) -> ValueSample[R]:
        return ValueSample(self.value)

    def get_choices(self) -> ValChm:
        return ValChm(self.value)


################
# Distribution #
################


SupportedConstraints = (
    EmptyConstraint
    | EqualityConstraint
    | MaskedConstraint["SupportedConstraints", ValueSample]
    | ChoiceMapConstraint
)

SupportedIncrementalConstraints = (
    EmptyConstraint
    | EqualityConstraint
    | MaskedConstraint["SupportedIncrementalConstraints", ValueSample]
    | ChoiceMapConstraint
)

SupportedGeneralConstraints = (
    EmptyConstraint
    | EqualityConstraint
    | MaskedConstraint["SupportedGeneralConstraints", ValueSample]
    | ChoiceMapConstraint
)


SupportedProjections = (
    EmptyProjection[ValueSample]
    | IdentityProjection[ValueSample]
    | SelectionProjection[ValueSample]
)


class Distribution(
    Generic[A, R],
    Simulateable[DistributionTrace[A, R], A, ValueSample[R], R],
    Assessable[DistributionTrace[A, R], A, ValueSample[R], R],
    ImportanceRequest[
        EmptyTrace["Distribution"], DistributionTrace[A, R]
    ].SupportsImportance[
        DistributionTrace[A, R],
        SupportedConstraints,
        A,
        ValueSample[R],
        R,
    ],
    ProjectRequest.SupportsProject,
    GeneralUpdateRequest[DistributionTrace[A, R],].UseAsDefaultUpdate[
        DistributionTrace[A, R],
        SupportedGeneralConstraints,
        SupportedGeneralConstraints,
        A,
        ValueSample[R],
        R,
    ],
    # TODO: fill in logic of incremental updates.
    # IncrementalRequest.SupportsIncrementalUpdate[
    #    SupportedIncrementalConstraints,
    #    ValueSample[R],
    #    R,
    # ],
    GeneralRegenerateRequest[
        DistributionTrace[A, R],
        DistributionTrace[A, R],
    ].SupportsGeneralRegenerate[
        DistributionTrace[A, R],
        DistributionTrace[A, R],
        A,
        ValueSample[R],
        R,
        SupportedProjections,
    ],
    GenerativeFunction[DistributionTrace[A, R], A, ValueSample[R], R],
):
    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *arguments,
    ) -> tuple[Score, R]:
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: R,
        *arguments,
    ) -> Weight:
        pass

    def simulate(
        self,
        key: PRNGKey,
        arguments: A,
    ) -> DistributionTrace[A, R]:
        (w, v) = self.random_weighted(key, *arguments)
        tr = DistributionTrace(self, arguments, v, w)
        return tr

    def importance_update(
        self,
        key: PRNGKey,
        constraint: SupportedConstraints,
        arguments: A,
    ) -> tuple[DistributionTrace[A, R], Weight, Projection]:
        match constraint:
            case EmptyConstraint():
                tr = self.simulate(key, arguments)
                weight = 0.0
                return tr, jnp.array(weight), EmptyProjection()

            case EqualityConstraint(v):
                w = self.estimate_logpdf(key, v, *arguments)
                tr = DistributionTrace(self, arguments, v, w)
                return tr, w, IdentityProjection()

            case MaskedConstraint(flag, subconstraint):
                # If it is valid.
                im_tr, weight, bwd = self.importance_update(
                    key, subconstraint, arguments
                )
                # If it is not.
                sim_tr, empty_weight, empty_bwd = self.importance_update(
                    key, EmptyConstraint(), arguments
                )

                tr, w = jtu.tree_map(
                    lambda v1, v2: jnp.where(flag, v1, v2),
                    (im_tr, weight),
                    (sim_tr, empty_weight),
                )

                # TODO: how to handle this?
                bwd_request = OrElseRequest(flag, bwd, empty_bwd)
                return (tr, w, bwd_request)

            # Compatibility with old usage:
            # here, we allow users to utilize choice maps to specify constraints
            # but only a few types.
            case ChoiceMapConstraint(choice_map):
                inner_constraint = constraint.get_value()
                return self.importance_update(key, inner_constraint, arguments)

    def incremental_update(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: SupportedIncrementalConstraints,
        argdiffs: Argdiffs,
    ) -> tuple[Trace, Weight, Retdiff, UpdateRequest]:
        match constraint:
            case EmptyConstraint():
                weight = 0.0
                return (
                    trace,
                    jnp.array(weight),
                    Diff.no_change(trace.get_retval()),
                    EmptyUpdateRequest(),
                )

            case EqualityConstraint(v):
                primals = Diff.tree_primal(argdiffs)
                old_score = trace.get_score()
                w = self.estimate_logpdf(key, v, *primals)
                inc_w = w - old_score
                new_tr = DistributionTrace(self, primals, v, w)
                return new_tr, inc_w, Diff.unknown_change(v), None

            case MaskedConstraint(flag, subconstraint):
                raise NotImplementedError

            # Compatibility with old usage:
            # here, we allow users to utilize choice maps to specify constraints
            # but only a few types.
            case ChoiceMapConstraint(choice_map):
                if isinstance(choice_map, EmptyChm):
                    constraint = EmptyConstraint()
                    return self.incremental_update(key, trace, constraint, argdiffs)
                elif isinstance(choice_map, ValChm):
                    constraint = EqualityConstraint(choice_map.get_value())
                    return self.incremental_update(key, trace, constraint, argdiffs)
                else:
                    raise NotImplementedError

    def general_update(
        self,
        key: PRNGKey,
        trace: DistributionTrace[A, R],
        constraint: SupportedGeneralConstraints,
        arguments: A,
    ) -> tuple[DistributionTrace[A, R], Weight, SupportedGeneralConstraints]:
        match constraint:
            case EmptyConstraint():
                old_score = trace.get_score()
                v = trace.get_retval()
                w = self.estimate_logpdf(key, v, *arguments)
                inc_w = w - old_score
                new_tr = DistributionTrace(self, arguments, v, w)
                return new_tr, inc_w, EmptyConstraint()

            case EqualityConstraint(v):
                old_score = trace.get_score()
                w = self.estimate_logpdf(key, v, *arguments)
                inc_w = w - old_score
                old_value = trace.get_retval()
                new_tr = DistributionTrace(self, arguments, v, w)
                return new_tr, inc_w, EqualityConstraint(old_value)

            case MaskedConstraint(flag, subconstraint):
                raise NotImplementedError

            # Compatibility with old usage:
            # here, we allow users to utilize choice maps to specify constraints
            # but only a few types.
            case ChoiceMapConstraint(choice_map):
                if isinstance(choice_map, EmptyChm):
                    constraint = EmptyConstraint()
                    return self.general_update(key, trace, constraint, arguments)
                elif isinstance(choice_map, ValChm):
                    constraint = EqualityConstraint(choice_map.get_value())
                    return self.general_update(key, trace, constraint, arguments)
                else:
                    raise NotImplementedError

    def general_regenerate(
        self,
        key: PRNGKey,
        trace: Trace,
        projection: SupportedProjections,
        arguments: Arguments,
    ) -> tuple[Trace, Weight, Sample]:
        match projection:
            case EmptyProjection():
                old_score = trace.get_score()
                v = trace.get_retval()
                w = self.estimate_logpdf(key, v, *arguments)
                inc_w = w - old_score
                old_value = trace.get_retval()
                new_tr = DistributionTrace(self, arguments, v, w)
                return new_tr, inc_w, EmptySample()

            case IdentityProjection():
                old_score = trace.get_score()
                w, v = self.random_weighted(key, *arguments)
                inc_w = w - old_score
                old_value = trace.get_retval()
                new_tr = DistributionTrace(self, arguments, v, w)
                return new_tr, inc_w, ValueSample(old_value)

            # Compatibility with old usage:
            case SelectionProjection(selection):
                raise NotImplementedError

    def project_update(
        self,
        key: PRNGKey,
        trace: Trace,
        projection: SupportedProjections,
    ) -> tuple[Weight, Constraint]:
        sample = trace.get_sample()
        projected = projection.project(sample)
        match projected:
            case EmptySample():
                return jnp.array(0.0), EmptyConstraint()

            case ValueSample(v):
                weight = trace.get_score()
                return weight, EqualityConstraint(v)

            case MaskedSample(v):
                raise NotImplementedError

    def assess(
        self,
        key: PRNGKey,
        sample: ValueSample
        | ValChm
        | ChoiceMapSample,  # TODO: the ValChm here is a type of paving over, to allow people to continue to use what they are used to.
        arguments: A,
    ):
        match sample:
            case ValChm(v):
                return self.assess(key, ValueSample(v), arguments)
            case ChoiceMapSample():
                key = jax.random.PRNGKey(0)
                v = sample.get_value()
                return self.assess(key, v, arguments)
            case ValueSample():
                key = jax.random.PRNGKey(0)
                v = sample.val
                w = self.estimate_logpdf(key, v, *arguments)
                return w, v


################
# ExactDensity #
################


class ExactDensity(Distribution):
    @abstractmethod
    def sample(self, key: PRNGKey, *arguments):
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, v: Retval, *arguments):
        raise NotImplementedError

    def __abstract_call__(self, *arguments):
        key = jax.random.PRNGKey(0)
        return self.sample(key, *arguments)

    def handle_kwargs(self) -> GenerativeFunction:
        @Pytree.partial(self)
        def sample_with_kwargs(self, key, arguments, kwargs):
            return self.sample(key, *arguments, **kwargs)

        @Pytree.partial(self)
        def logpdf_with_kwargs(self, v, arguments, kwargs):
            return self.logpdf(v, *arguments, **kwargs)

        return ExactDensityFromCallables(
            sample_with_kwargs,
            logpdf_with_kwargs,
        )

    def random_weighted(
        self,
        key: PRNGKey,
        *arguments,
    ) -> tuple[Score, Retval]:
        """Given arguments to the distribution, sample from the distribution,
        and return the exact log density of the sample, and the sample."""
        v = self.sample(key, *arguments)
        w = self.estimate_logpdf(key, v, *arguments)
        return (w, v)

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *arguments,
    ) -> Weight:
        """Given a sample and arguments to the distribution, return the exact
        log density of the sample."""
        w = self.logpdf(v, *arguments)
        if w.shape:
            return jnp.sum(w)
        else:
            return w


@Pytree.dataclass
class ExactDensityFromCallables(ExactDensity):
    sampler: Closure
    logpdf_evaluator: Closure

    def sample(self, key, *arguments):
        return self.sampler(key, *arguments)

    def logpdf(self, v, *arguments):
        return self.logpdf_evaluator(v, *arguments)


def exact_density(
    sample: Callable[..., Any],
    logpdf: Callable[..., Any],
):
    if not isinstance(sample, Closure):
        sample = Pytree.partial()(sample)

    if not isinstance(logpdf, Closure):
        logpdf = Pytree.partial()(logpdf)

    return ExactDensityFromCallables(sample, logpdf)
