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

from genjax._src.core.generative import (
    Arguments,
    ChoiceMap,
    ChoiceMapConstraint,
    ChoiceMapEditRequest,
    ChoiceMapProjection,
    ChoiceMapSample,
    EmptyConstraint,
    EmptySample,
    EqualityConstraint,
    GenerativeFunction,
    IdentityProjection,
    Mask,
    MaskedConstraint,
    MaskedSample,
    Retdiff,
    Retval,
    Sample,
    SampleCoercableToChoiceMap,
    Score,
    Selection,
    SelectionProjection,
    SelectionRegenerateRequest,
    Trace,
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
    overload,
    tuple,
)

A = TypeVar("A", bound=Arguments)
R = TypeVar("R", bound=Retval)

#####################
# DistributionTrace #
#####################


@Pytree.dataclass
class DistributionTrace(
    Generic[A, R],
    SampleCoercableToChoiceMap,
    Trace["Distribution", A, "ValueSample[R]", R],
):
    gen_fn: "Distribution"
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


class Distribution(
    Generic[A, R],
    GenerativeFunction[
        DistributionTrace[A, R],
        A,
        ValueSample[R],
        R,
        ChoiceMapConstraint[EmptyConstraint | EqualityConstraint[R | Mask[R]]],
        SelectionProjection | ChoiceMapProjection,
        ChoiceMapEditRequest[A] | SelectionRegenerateRequest[A],
    ],
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

    def assess(
        self,
        key: PRNGKey,
        # TODO: the ValChm here is a type of paving over, to allow people to continue to use what they are used to.
        sample: ValChm | ChoiceMapSample[ValueSample | MaskedSample] | ValueSample,
        arguments: A,
    ) -> tuple[Score | Mask[Score], R | Mask[R]]:
        match sample:
            case ValChm(v):
                return self.assess(key, ValueSample(v), arguments)

            case ChoiceMapSample():
                v: Sample = sample.get_value()
                match v:
                    case MaskedSample(flag, sample_value):
                        score, return_value = self.assess(key, sample_value, arguments)
                        return Mask.maybe(flag, score), Mask.maybe(flag, return_value)
                    case ValueSample():
                        return self.assess(key, v, arguments)

            case ValueSample():
                v = sample.get_value()
                match v:
                    case Mask(flag, value):
                        w = self.estimate_logpdf(key, value, *arguments)
                        return Mask(flag, w), Mask(flag, value)
                    case _:
                        w = self.estimate_logpdf(key, v, *arguments)
                        return w, v

    def importance_edit(
        self,
        key: PRNGKey,
        constraint: ChoiceMapConstraint[
            EmptyConstraint | EqualityConstraint[R | Mask[R]]
        ],
        arguments: A,
    ) -> tuple[DistributionTrace[A, R], Weight, ChoiceMapProjection]:
        inner_constraint = constraint.get_value()
        match inner_constraint:
            case EmptyConstraint():
                tr = self.simulate(key, arguments)
                weight = 0.0
                return (
                    tr,
                    jnp.array(weight),
                    ChoiceMapProjection(ChoiceMap.value(EmptyProjection())),
                )

            case EqualityConstraint(v):
                if isinstance(v, Mask):

                    def true_branch(key, value, args):
                        w = self.estimate_logpdf(key, value, *args)
                        return w, value

                    def false_branch(key, value, args):
                        _, v = self.random_weighted(key, *args)
                        return jnp.array(0.0), v

                    w, value = jax.lax.cond(
                        v.flag, true_branch, false_branch, key, v.value, arguments
                    )
                    tr = DistributionTrace(self, arguments, value, w)
                    return (
                        tr,
                        w,
                        ChoiceMapProjection(
                            ChoiceMap.maybe(
                                v.flag, ChoiceMap.value(IdentityProjection())
                            )
                        ),
                    )
                else:
                    w = self.estimate_logpdf(key, v, *arguments)
                    tr = DistributionTrace(self, arguments, v, w)
                    return (
                        tr,
                        w,
                        ChoiceMapProjection(ChoiceMap.value(IdentityProjection())),
                    )

            case MaskedConstraint(flag, subconstraint):
                raise NotImplementedError

    def project_edit(
        self,
        key: PRNGKey,
        trace: DistributionTrace[A, R],
        projection: ChoiceMapProjection[EmptySample | ValueSample[R]]
        | SelectionProjection[EmptySample | ValueSample[R]],
    ) -> tuple[Weight, ChoiceMapConstraint]:
        sample = trace.get_choices()
        projected = projection.project(ChoiceMapSample(sample))
        value = projected.get_value()
        match value:
            case EmptySample():
                return jnp.array(0.0), ChoiceMapConstraint(ChoiceMap.empty())

            case ValueSample(v):
                weight = trace.get_score()
                return weight, ChoiceMapConstraint(ChoiceMap.value(v))

            case MaskedSample(v):
                raise NotImplementedError

    def choice_map_edit(
        self,
        key: PRNGKey,
        trace: DistributionTrace[A, R],
        constraint: ChoiceMapConstraint[EmptyConstraint | EqualityConstraint[R]],
        arguments: A,
    ) -> tuple[DistributionTrace[A, R], Weight, ChoiceMapConstraint]:
        value = constraint.get_value()
        match value:
            case EmptyConstraint():
                old_score = trace.get_score()
                v = trace.get_retval()
                w = self.estimate_logpdf(key, v, *arguments)
                inc_w = w - old_score
                new_tr = DistributionTrace(self, arguments, v, w)
                return new_tr, inc_w, ChoiceMapConstraint(ChoiceMap.empty())

            case EqualityConstraint(v):
                if isinstance(v, Mask):
                    flag, value = v.flag, v.value

                    def true_branch(key, tr, args):
                        new_tr, inc_w, c = self.choice_map_edit(
                            key,
                            tr,
                            ChoiceMapConstraint(
                                ChoiceMap.value(EqualityConstraint(v.value))
                            ),
                            args,
                        )
                        return new_tr, inc_w

                    def false_branch(key, tr, args):
                        new_tr, inc_w, c = self.choice_map_edit(
                            key, tr, ChoiceMapConstraint(ChoiceMap.empty()), args
                        )
                        return new_tr, inc_w

                    new_tr, inc_w = jax.lax.cond(
                        flag, true_branch, false_branch, key, trace, arguments
                    )
                    shared_constraint = ChoiceMapConstraint(
                        ChoiceMap.maybe(flag, ChoiceMap.value(trace.get_retval()))
                    )
                    return new_tr, inc_w, shared_constraint
                else:
                    old_score = trace.get_score()
                    w = self.estimate_logpdf(key, v, *arguments)
                    inc_w = w - old_score
                    old_value = trace.get_retval()
                    new_tr = DistributionTrace(self, arguments, v, w)
                    return (
                        new_tr,
                        inc_w,
                        ChoiceMapConstraint(
                            ChoiceMap.value(EqualityConstraint(old_value))
                        ),
                    )

    def selection_regenerate_edit(
        self,
        key: PRNGKey,
        trace: Trace,
        selection: Selection,
        arguments: Arguments,
    ) -> tuple[
        Trace,
        Weight,
        ChoiceMapConstraint[EmptyConstraint | EqualityConstraint[R]],
    ]:
        match selection.check():
            case True:
                new_score, new_value = self.random_weighted(key, *arguments)
                old_score = trace.get_score()
                old_value = trace.get_retval()
                return (
                    DistributionTrace(self, arguments, new_value, new_score),
                    new_score - old_score,
                    ChoiceMapConstraint(ChoiceMap.value(old_value)),
                )

            case False:
                return trace, jnp.array(0.0), ChoiceMapConstraint(ChoiceMap.empty())

            case BoolArray:

                def true_branch(key, tr, args):
                    new_tr, inc_w, c = self.selection_regenerate_edit(
                        key,
                        tr,
                        Selection.empty(),
                        args,
                    )
                    return new_tr, inc_w

                def false_branch(key, tr, args):
                    new_tr, inc_w, c = self.selection_regenerate_edit(
                        key, tr, Selection.all(), args
                    )
                    return new_tr, inc_w

                flag = selection.check()
                new_tr, inc_w = jax.lax.cond(
                    flag, true_branch, false_branch, key, trace, arguments
                )
                shared_constraint = ChoiceMapConstraint(
                    ChoiceMap.maybe(flag, ChoiceMap.value(trace.get_retval()))
                )
                return new_tr, inc_w, shared_constraint

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        request: SelectionRegenerateRequest[A],
        arguments: A,
    ) -> tuple[DistributionTrace, Weight, Retdiff, ChoiceMapEditRequest[A]]:
        pass

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        request: ChoiceMapEditRequest[A],
        arguments: A,
    ) -> tuple[DistributionTrace, Weight, Retdiff, ChoiceMapEditRequest[A]]:
        pass

    def edit(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        request: ChoiceMapEditRequest[A] | SelectionRegenerateRequest[A],
        arguments: A,
    ) -> tuple[
        DistributionTrace,
        Weight,
        Retdiff,
        ChoiceMapEditRequest[A] | SelectionRegenerateRequest[A],
    ]:
        match request:
            case ChoiceMapEditRequest(chm_constraint):
                new_trace, weight, discard_chm = self.choice_map_edit(
                    key, trace, chm_constraint, arguments
                )
                original_arguments = trace.get_args()
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    ChoiceMapEditRequest(discard_chm),
                )

            case SelectionRegenerateRequest(projection):
                new_trace, weight, bwd_choice_map_constraint = (
                    self.selection_regenerate_edit(key, trace, projection, arguments)
                )
                original_arguments = trace.get_args()
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    ChoiceMapEditRequest(bwd_choice_map_constraint),
                )


################
# ExactDensity #
################


class ExactDensity(Distribution):
    def __abstract_call__(self, *arguments):
        key = jax.random.PRNGKey(0)
        return self.sample(key, *arguments)

    @abstractmethod
    def sample(self, key: PRNGKey, *arguments):
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, v: Retval, *arguments):
        raise NotImplementedError

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
