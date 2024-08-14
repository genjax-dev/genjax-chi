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
    Masked,
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
    args: A
    value: R
    score: Score

    def get_args(self) -> A:
        return self.args

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
        A,
        ValueSample[R],
        R,
        ChoiceMapConstraint[EmptyConstraint | EqualityConstraint[R | Masked[R]]],
        SelectionProjection | ChoiceMapProjection,
        ChoiceMapEditRequest | SelectionRegenerateRequest,
    ],
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

    def simulate(
        self,
        key: PRNGKey,
        args: A,
    ) -> DistributionTrace[A, R]:
        (w, v) = self.random_weighted(key, *args)
        tr = DistributionTrace(self, args, v, w)
        return tr

    def assess(
        self,
        key: PRNGKey,
        # TODO: the ValChm here is a type of paving over, to allow people to continue to use what they are used to.
        sample: ValChm | ChoiceMapSample[ValueSample | MaskedSample] | ValueSample,
        args: A,
    ) -> tuple[Score, R]:
        match sample:
            case ValChm(v):
                return self.assess(key, ValueSample(v), args)

            case ChoiceMapSample():
                v: Sample = sample.get_value()
                match v:
                    case MaskedSample(flag, sample_value):
                        score, return_value = self.assess(key, sample_value, args)
                        return jnp.where(flag, score, -jnp.inf), return_value
                    case ValueSample():
                        return self.assess(key, v, args)

            case ValueSample():
                v = sample.get_value()
                match v:
                    case Masked(flag, value):
                        w = self.estimate_logpdf(key, value, *args)
                        return jnp.where(flag, w, -jnp.inf), value
                    case _:
                        w = self.estimate_logpdf(key, v, *args)
                        return w, v

    def importance_edit(
        self,
        key: PRNGKey,
        constraint: ChoiceMapConstraint[
            EmptyConstraint | EqualityConstraint[R | Masked[R]]
        ],
        args: A,
    ) -> tuple[DistributionTrace[A, R], Weight, ChoiceMapProjection]:
        inner_constraint = constraint.get_value()
        match inner_constraint:
            case EmptyConstraint():
                tr = self.simulate(key, args)
                weight = 0.0
                return (
                    tr,
                    jnp.array(weight),
                    ChoiceMapProjection(ChoiceMap.value(EmptyProjection())),
                )

            case EqualityConstraint(v):
                if isinstance(v, Masked):

                    def true_branch(key, value, args):
                        w = self.estimate_logpdf(key, value, *args)
                        return w, value

                    def false_branch(key, value, args):
                        _, v = self.random_weighted(key, *args)
                        return jnp.array(0.0), v

                    w, value = jax.lax.cond(
                        v.flag, true_branch, false_branch, key, v.value, args
                    )
                    tr = DistributionTrace(self, args, value, w)
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
                    w = self.estimate_logpdf(key, v, *args)
                    tr = DistributionTrace(self, args, v, w)
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
        args: A,
    ) -> tuple[DistributionTrace[A, R], Weight, ChoiceMapConstraint]:
        value = constraint.get_value()
        match value:
            case EmptyConstraint():
                old_score = trace.get_score()
                v = trace.get_retval()
                w = self.estimate_logpdf(key, v, *args)
                inc_w = w - old_score
                new_tr = DistributionTrace(self, args, v, w)
                return new_tr, inc_w, ChoiceMapConstraint(ChoiceMap.empty())

            case EqualityConstraint(v):
                if isinstance(v, Masked):
                    flag, value = v.flag, v.value

                    def true_branch(key, tr, args):
                        new_tr, inc_w, _c = self.choice_map_edit(
                            key,
                            tr,
                            ChoiceMapConstraint(
                                ChoiceMap.value(EqualityConstraint(v.value))
                            ),
                            args,
                        )
                        return new_tr, inc_w

                    def false_branch(key, tr, args):
                        new_tr, inc_w, _c = self.choice_map_edit(
                            key, tr, ChoiceMapConstraint(ChoiceMap.empty()), args
                        )
                        return new_tr, inc_w

                    new_tr, inc_w = jax.lax.cond(
                        flag, true_branch, false_branch, key, trace, args
                    )
                    shared_constraint = ChoiceMapConstraint(
                        ChoiceMap.maybe(flag, ChoiceMap.value(trace.get_retval()))
                    )
                    return new_tr, inc_w, shared_constraint
                else:
                    old_score = trace.get_score()
                    w = self.estimate_logpdf(key, v, *args)
                    inc_w = w - old_score
                    old_value = trace.get_retval()
                    new_tr = DistributionTrace(self, args, v, w)
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
        args: Arguments,
    ) -> tuple[
        Trace,
        Weight,
        ChoiceMapConstraint[EmptyConstraint | EqualityConstraint[R]],
    ]:
        match selection.check():
            case True:
                new_score, new_value = self.random_weighted(key, *args)
                old_score = trace.get_score()
                old_value = trace.get_retval()
                return (
                    DistributionTrace(self, args, new_value, new_score),
                    new_score - old_score,
                    ChoiceMapConstraint(ChoiceMap.value(old_value)),
                )

            case False:
                return trace, jnp.array(0.0), ChoiceMapConstraint(ChoiceMap.empty())

            case BoolArray:

                def true_branch(key, tr, args):
                    new_tr, inc_w, _c = self.selection_regenerate_edit(
                        key,
                        tr,
                        Selection.empty(),
                        args,
                    )
                    return new_tr, inc_w

                def false_branch(key, tr, args):
                    new_tr, inc_w, _c = self.selection_regenerate_edit(
                        key, tr, Selection.all(), args
                    )
                    return new_tr, inc_w

                flag = selection.check()
                new_tr, inc_w = jax.lax.cond(
                    flag, true_branch, false_branch, key, trace, args
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
        request: SelectionRegenerateRequest,
        args: A,
    ) -> tuple[DistributionTrace, Weight, Retdiff, ChoiceMapEditRequest]:
        pass

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        request: ChoiceMapEditRequest,
        args: A,
    ) -> tuple[DistributionTrace, Weight, Retdiff, ChoiceMapEditRequest]:
        pass

    def edit(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        request: ChoiceMapEditRequest | SelectionRegenerateRequest,
        args: A,
    ) -> tuple[
        DistributionTrace,
        Weight,
        Retdiff,
        ChoiceMapEditRequest | SelectionRegenerateRequest,
    ]:
        match request:
            case ChoiceMapEditRequest(chm_constraint):
                new_trace, weight, discard_chm = self.choice_map_edit(
                    key, trace, chm_constraint, args
                )
                trace.get_args()
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    ChoiceMapEditRequest(discard_chm),
                )

            case SelectionRegenerateRequest(projection):
                new_trace, weight, bwd_choice_map_constraint = (
                    self.selection_regenerate_edit(key, trace, projection, args)
                )
                trace.get_args()
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
    def __abstract_call__(self, *args):
        key = jax.random.PRNGKey(0)
        return self.sample(key, *args)

    @abstractmethod
    def sample(self, key: PRNGKey, *args):
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, v: Retval, *args):
        raise NotImplementedError

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
        """Given arguments to the distribution, sample from the distribution,
        and return the exact log density of the sample, and the sample."""
        v = self.sample(key, *args)
        w = self.estimate_logpdf(key, v, *args)
        return (w, v)

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *args,
    ) -> Weight:
        """Given a sample and arguments to the distribution, return the exact
        log density of the sample."""
        w = self.logpdf(v, *args)
        if w.shape:
            return jnp.sum(w)
        else:
            return w


@Pytree.dataclass
class ExactDensityFromCallables(ExactDensity):
    sampler: Closure
    logpdf_evaluator: Closure

    def sample(self, key, *args):
        return self.sampler(key, *args)

    def logpdf(self, v, *args):
        return self.logpdf_evaluator(v, *args)


def exact_density(
    sample: Callable[..., Any],
    logpdf: Callable[..., Any],
):
    if not isinstance(sample, Closure):
        sample = Pytree.partial()(sample)

    if not isinstance(logpdf, Closure):
        logpdf = Pytree.partial()(logpdf)

    return ExactDensityFromCallables(sample, logpdf)
