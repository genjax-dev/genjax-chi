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
from jax.experimental import checkify

from genjax._src.checkify import optional_check
from genjax._src.core.generative import (
    Argdiffs,
    Arguments,
    ChoiceMap,
    Constraint,
    EmptyConstraint,
    EmptyUpdateRequest,
    EqualityConstraint,
    GenerativeFunction,
    ImportanceRequest,
    IncrementalRequest,
    Mask,
    MaskedConstraint,
    MaskedUpdateRequest,
    ProjectRequest,
    Retdiff,
    Retval,
    Sample,
    Score,
    Trace,
    UpdateRequest,
    ValueSample,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import staged_check
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    PRNGKey,
    Tuple,
    TypeVar,
    static_check_is_concrete,
    typecheck,
)

#####
# DistributionTrace
#####


@Pytree.dataclass
class DistributionTrace(
    Trace,
):
    gen_fn: GenerativeFunction
    args: Tuple
    value: Any
    score: FloatArray

    def get_args(self) -> Tuple:
        return self.args

    def get_retval(self) -> Any:
        return self.value

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_score(self) -> FloatArray:
        return self.score

    def get_sample(self) -> ChoiceMap:
        return ChoiceMap.value(self.value)


################
# Distribution #
################

R = TypeVar("R")

SupportedConstraints = EmptyConstraint | EqualityConstraint 

class Distribution(
    Generic[R],
    IncrementalRequest.SupportsIncrementalUpdate,
    ImportanceRequest[SupportedConstraints].SupportsImportance,
    ProjectRequest.SupportsProject,
    GenerativeFunction[ValueSample[R], R],
):
    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Tuple[Score, R]:
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
        args: Tuple,
    ) -> Trace:
        (w, v) = self.random_weighted(key, *args)
        tr = DistributionTrace(self, args, v, w)
        return tr

    def importance(
        self,
        key: PRNGKey,
        constraint: EqualityConstraint,
        args: Arguments,
    ) -> Tuple[Trace, Weight]:
        raise NotImplementedError
    
    def update_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        primals = Diff.tree_primal(argdiffs)
        match constraint:
            case EmptyConstraint():
                old_sample = trace.get_choices()
                old_retval = trace.get_retval()
                new_score, _ = self.assess(old_sample, primals)
                new_trace = DistributionTrace(
                    self, primals, old_sample.get_value(), new_score
                )
                return (
                    new_trace,
                    new_score - trace.get_score(),
                    Diff.tree_diff_no_change(old_retval),
                    EmptyUpdateRequest(),
                )

            case MaskedConstraint(flag, problem):
                if staged_check(flag):
                    return self.update(
                        key, trace, IncrementalRequest(argdiffs, problem)
                    )
                else:
                    return self.update_constraint_masked_constraint(
                        key, trace, constraint, argdiffs
                    )

            case ChoiceMap():
                check = constraint.has_value()
                v = constraint.get_value()
                if isinstance(v, UpdateRequest):
                    return self.update(key, trace, IncrementalRequest(argdiffs, v))
                elif static_check_is_concrete(check) and check:
                    fwd = self.estimate_logpdf(key, v, *primals)
                    bwd = trace.get_score()
                    w = fwd - bwd
                    new_tr = DistributionTrace(self, primals, v, fwd)
                    discard = trace.get_sample()
                    retval_diff = Diff.tree_diff_unknown_change(v)
                    return (
                        new_tr,
                        w,
                        retval_diff,
                        discard,
                    )
                elif static_check_is_concrete(check):
                    value_chm = trace.get_choices()
                    v = value_chm.get_value()
                    fwd = self.estimate_logpdf(key, v, *primals)
                    bwd = trace.get_score()
                    w = fwd - bwd
                    new_tr = DistributionTrace(self, primals, v, fwd)
                    retval_diff = Diff.tree_diff_no_change(v)
                    return (new_tr, w, retval_diff, EmptyUpdateRequest())
                else:
                    # Whether or not the choice map has a value is dynamic...
                    # We must handled with a cond.
                    def _true_branch(key, new_value, old_value):
                        fwd = self.estimate_logpdf(key, new_value, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        return (new_value, w, fwd)

                    def _false_branch(key, new_value, old_value):
                        fwd = self.estimate_logpdf(key, old_value, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        return (old_value, w, fwd)

                    masked_value: Mask = v
                    flag = masked_value.flag
                    new_value = masked_value.value
                    old_value = trace.get_choices().get_value()

                    new_value, w, score = jax.lax.cond(
                        flag,
                        _true_branch,
                        _false_branch,
                        key,
                        new_value,
                        old_value,
                    )
                    return (
                        DistributionTrace(self, primals, new_value, score),
                        w,
                        Diff.tree_diff_unknown_change(new_value),
                        MaskedUpdateRequest(flag, old_value),
                    )

            case _:
                raise Exception("Unhandled constraint in update.")

    @typecheck
    def incremental_update(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        return self.update_constraint(key, trace, update_request, argdiffs)

    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
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
    ) -> Tuple[Score, Retval]:
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
        sample: ValueSample,
        args: Tuple,
    ):
        key = jax.random.PRNGKey(0)
        v = sample.get_value()
        match v:
            case Mask(flag, value):

                def _check():
                    check_flag = jnp.all(flag)
                    checkify.check(
                        check_flag,
                        "Attempted to unmask when a mask flag is False: the masked value is invalid.\n",
                    )

                optional_check(_check)
                w = self.estimate_logpdf(key, value, *args)
                return w, value
            case _:
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
