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
    ChoiceMapConstraint,
    EmptyChm,
    EmptyConstraint,
    EmptySample,
    EmptyUpdateRequest,
    EqualityConstraint,
    GeneralRegenerateRequest,
    GeneralUpdateRequest,
    GenerativeFunction,
    IdentityProjection,
    ImportanceRequest,
    MaskedConstraint,
    ProjectRequest,
    Retdiff,
    Retval,
    Sample,
    Score,
    Trace,
    UpdateRequest,
    ValChm,
    ValueSample,
    Weight,
)
from genjax._src.core.generative.core import (
    EmptyProjection,
    SelectionProjection,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    PRNGKey,
    TypeVar,
    tuple,
    typecheck,
)

R = TypeVar("R")

#####
# DistributionTrace
#####


@Pytree.dataclass
class DistributionTrace(
    Generic[R],
    Trace["Distribution[R]", "ValueSample"],
):
    gen_fn: "Distribution"
    args: tuple
    value: Any
    score: FloatArray

    def get_args(self) -> tuple:
        return self.args

    def get_retval(self) -> R:
        return self.value

    def get_gen_fn(self) -> "Distribution[R]":
        return self.gen_fn

    def get_score(self) -> Score:
        return self.score

    def get_sample(self) -> ValueSample:
        return ValueSample(self.value)

    def get_choices(self) -> ValChm:
        return ValChm(self.value)


################
# Distribution #
################


SupportedConstraints = (
    EmptyConstraint
    | EqualityConstraint
    | MaskedConstraint[ValueSample, "SupportedConstraints"]
    | ChoiceMapConstraint
)

SupportedIncrementalConstraints = (
    EmptyConstraint
    | EqualityConstraint
    | MaskedConstraint[ValueSample, "SupportedIncrementalConstraints"]
    | ChoiceMapConstraint
)

SupportedGeneralConstraints = (
    EmptyConstraint
    | EqualityConstraint
    | MaskedConstraint[ValueSample, "SupportedGeneralConstraints"]
    | ChoiceMapConstraint
)

SupportedProjections = EmptyProjection | IdentityProjection | SelectionProjection


class Distribution(
    Generic[R],
    GeneralUpdateRequest.SupportsGeneralUpdate[
        SupportedGeneralConstraints,
        ValueSample[R],
        R,
    ],
    # IncrementalRequest.SupportsIncrementalUpdate[
    #    SupportedIncrementalConstraints,
    #    ValueSample[R],
    #    R,
    # ],
    GeneralRegenerateRequest.SupportsGeneralRegenerate[
        SupportedProjections,
        ValueSample[R],
        R,
    ],
    ImportanceRequest.SupportsImportanceUpdate[
        SupportedConstraints,
        ValueSample[R],
        R,
    ],
    ProjectRequest.SupportsProjectUpdate,
    GenerativeFunction[ValueSample[R], R],
):
    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> tuple[Score, R]:
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: R,
        *args,
    ) -> Weight:
        pass

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Trace:
        (w, v) = self.random_weighted(key, *args)
        tr = DistributionTrace(self, args, v, w)
        return tr

    def importance_update(
        self,
        key: PRNGKey,
        constraint: SupportedConstraints,
        args: Arguments,
    ) -> tuple[Trace, Weight, UpdateRequest]:
        match constraint:
            case EmptyConstraint():
                tr = self.simulate(key, args)
                weight = 0.0
                return tr, jnp.array(weight), ProjectRequest(IdentityProjection())

            case EqualityConstraint(v):
                w = self.estimate_logpdf(key, v, *args)
                tr = DistributionTrace(self, args, v, w)
                return tr, w, ProjectRequest(EmptyProjection())

            case MaskedConstraint(flag, subconstraint):
                # If it is valid.
                im_tr, weight, bwd = self.importance_update(key, subconstraint, args)
                # If it is not.
                sim_tr, empty_weight, empty_bwd = self.importance_update(
                    key, EmptyConstraint(), args
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
                if isinstance(choice_map, EmptyChm):
                    constraint = EmptyConstraint()
                    return self.importance_update(key, constraint, args)
                elif isinstance(choice_map, ValChm):
                    constraint = EqualityConstraint(choice_map.get_value())
                    return self.importance_update(key, constraint, args)
                else:
                    raise NotImplementedError

    @typecheck
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

    @typecheck
    def general_update(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: SupportedGeneralConstraints,
        arguments: Arguments,
    ) -> tuple[Trace, Weight, Retdiff, Sample]:
        match constraint:
            case EmptyConstraint():
                old_score = trace.get_score()
                v = trace.get_retval()
                w = self.estimate_logpdf(key, v, *arguments)
                inc_w = w - old_score
                new_tr = DistributionTrace(self, arguments, v, w)
                return new_tr, inc_w, Diff.unknown_change(v), EmptySample()

            case EqualityConstraint(v):
                old_score = trace.get_score()
                w = self.estimate_logpdf(key, v, *arguments)
                inc_w = w - old_score
                old_value = trace.get_retval()
                new_tr = DistributionTrace(self, arguments, v, w)
                return new_tr, inc_w, Diff.unknown_change(v), ValueSample(old_value)

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

    @typecheck
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
            case SelectionProjection():
                raise NotImplementedError

    @typecheck
    def project_update(
        self,
        key: PRNGKey,
        trace: Trace,
        projection: SupportedProjections,
    ) -> tuple[Weight, UpdateRequest]:
        raise NotImplementedError

    @typecheck
    def assess(
        self,
        sample: ValueSample,
        args: tuple,
    ):
        raise NotImplementedError


################
# ExactDensity #
################


class ExactDensity(Distribution):
    @abstractmethod
    def sample(self, key: PRNGKey, *args):
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, v: Retval, *args):
        raise NotImplementedError

    def __abstract_call__(self, *args):
        key = jax.random.PRNGKey(0)
        return self.sample(key, *args)

    def handle_kwargs(self) -> GenerativeFunction:
        @Pytree.partial(self)
        def sample_with_kwargs(self, key, args, kwargs):
            return self.sample(key, *args, **kwargs)

        @Pytree.partial(self)
        def logpdf_with_kwargs(self, v, args, kwargs):
            return self.logpdf(v, *args, **kwargs)

        return ExactDensityFromCallables(
            sample_with_kwargs,
            logpdf_with_kwargs,
        )

    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> tuple[Score, Retval]:
        """
        Given arguments to the distribution, sample from the distribution, and return the exact log density of the sample, and the sample.
        """
        v = self.sample(key, *args)
        w = self.estimate_logpdf(key, v, *args)
        return (w, v)

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *args,
    ) -> Weight:
        """
        Given a sample and arguments to the distribution, return the exact log density of the sample.
        """
        w = self.logpdf(v, *args)
        if w.shape:
            return jnp.sum(w)
        else:
            return w

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: ValueSample
        | ValChm,  # TODO: the ValChm here is a type of paving over, to allow people to continue to use what they are used to.
        args: tuple,
    ):
        match sample:
            case ValChm():
                key = jax.random.PRNGKey(0)
                v = sample.get_value()
                w = self.estimate_logpdf(key, v, *args)
                return w, v
            case ValueSample():
                key = jax.random.PRNGKey(0)
                v = sample.val
                w = self.estimate_logpdf(key, v, *args)
                return w, v


@Pytree.dataclass
class ExactDensityFromCallables(ExactDensity):
    sampler: Closure
    logpdf_evaluator: Closure

    def sample(self, key, *args):
        return self.sampler(key, *args)

    def logpdf(self, v, *args):
        return self.logpdf_evaluator(v, *args)


@typecheck
def exact_density(
    sample: Callable[..., Any],
    logpdf: Callable[..., Any],
):
    if not isinstance(sample, Closure):
        sample = Pytree.partial()(sample)

    if not isinstance(logpdf, Closure):
        logpdf = Pytree.partial()(logpdf)

    return ExactDensityFromCallables(sample, logpdf)
