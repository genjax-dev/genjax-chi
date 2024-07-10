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
from jax.lax import cond

from genjax._src.checkify import optional_check
from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    Constraint,
    EmptyConstraint,
    EmptyTrace,
    EmptyUpdateRequest,
    GenerativeFunction,
    ImportanceUpdateRequest,
    IncrementalUpdateRequest,
    Mask,
    MaskedConstraint,
    MaskedUpdateRequest,
    ProjectUpdateRequest,
    Retdiff,
    Retval,
    Sample,
    Score,
    Selection,
    Trace,
    UpdateRequest,
    Weight,
)
from genjax._src.core.generative.choice_map import ChoiceMapConstraint, ChoiceMapSample
from genjax._src.core.generative.core import Arguments, ConstraintUpdateRequest
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import staged_check
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Bool,
    BoolArray,
    Callable,
    Generic,
    PRNGKey,
    Tuple,
    TypeVar,
    dispatch,
    overload,
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
    args: Arguments
    value: Any
    score: Score

    def get_args(self) -> Arguments:
        return self.args

    def get_retval(self) -> Any:
        return self.value

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_score(self) -> Score:
        return self.score

    def get_sample(self) -> Sample:
        return ChoiceMapSample(ChoiceMap.value(self.value))


################
# Distribution #
################

R = TypeVar("R")


class Distribution(Generic[R], GenerativeFunction):
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

    def update_constraint_masked_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: MaskedConstraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        old_sample = trace.get_sample()

        def update_branch(key, trace, constraint, argdiffs):
            tr, w, rd, _ = self.update(
                key, trace, IncrementalUpdateRequest(argdiffs, constraint)
            )
            return (
                tr,
                w,
                rd,
                MaskedUpdateRequest(True, old_sample),
            )

        def do_nothing_branch(key, trace, constraint, argdiffs):
            tr, w, _, _ = self.update(
                key, trace, IncrementalUpdateRequest(argdiffs, EmptyUpdateRequest())
            )
            return (
                tr,
                w,
                Diff.tree_diff_unknown_change(tr.get_retval()),
                MaskedUpdateRequest(False, old_sample),
            )

        return jax.lax.cond(
            constraint.flag,
            update_branch,
            do_nothing_branch,
            key,
            trace,
            constraint.constraint,
            argdiffs,
        )

    def update_masked(
        self,
        key: PRNGKey,
        trace: Trace,
        flag: Bool | BoolArray,
        problem: UpdateRequest,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        old_value = trace.get_retval()
        primals = Diff.tree_primal(argdiffs)
        possible_trace, w, retdiff, bwd_problem = self.update(
            key,
            trace,
            IncrementalUpdateRequest(argdiffs, problem),
        )
        new_value = possible_trace.get_retval()
        w = w * flag
        bwd_problem = MaskedUpdateRequest(flag, bwd_problem)
        new_trace = DistributionTrace(
            self,
            primals,
            jax.lax.select(flag, new_value, old_value),
            jax.lax.select(flag, possible_trace.get_score(), trace.get_score()),
        )

        return new_trace, w, retdiff, bwd_problem

    def update_project(
        self,
        trace: Trace,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        original = trace.get_score()
        removed_value = trace.get_retval()
        retdiff = Diff.tree_diff_unknown_change(trace.get_retval())
        return (
            EmptyTrace(self),
            -original,
            retdiff,
            ChoiceMap.value(removed_value),
        )

    def update_selection_project(
        self,
        key: PRNGKey,
        trace: Trace,
        selection: Selection,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        check = () in selection

        return self.update(
            key,
            trace,
            IncrementalUpdateRequest(
                argdiffs,
                MaskedUpdateRequest.maybe(check, ProjectUpdateRequest()),
            ),
        )

    @overload
    def update_incremental(
        self,
        trace: Trace,
        request: EmptyUpdateRequest,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        sample = trace.get_sample()
        primals = Diff.tree_primal(argdiffs)
        new_score, _ = self.assess(sample, primals)
        new_trace = DistributionTrace(self, primals, sample.get_value(), new_score)
        return (
            new_trace,
            new_score - trace.get_score(),
            Diff.tree_diff_no_change(trace.get_retval()),
            EmptyUpdateRequest(),
        )

    @overload
    def update_incremental_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: MaskedConstraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        flag, subconstraint = constraint.mask, constraint.subconstraint
        if staged_check(flag):
            return self.update(
                key,
                trace,
                IncrementalUpdateRequest(
                    argdiffs, ConstraintUpdateRequest(subconstraint)
                ),
            )
        else:
            return self.update_incremental_constraint(key, trace, constraint, argdiffs)

    @overload
    def update_incremental_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: EmptyConstraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        primals = Diff.tree_primal(argdiffs)
        old_sample = trace.get_sample()
        old_retval = trace.get_retval()
        new_score, _ = self.assess(old_sample, primals)
        new_trace = DistributionTrace(self, primals, old_sample.get_value(), new_score)
        return (
            new_trace,
            new_score - trace.get_score(),
            Diff.tree_diff_no_change(old_retval),
            EmptyUpdateRequest(),
        )

    @overload
    def update_incremental_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: ChoiceMapConstraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        primals = Diff.tree_primal(argdiffs)
        check = constraint.has_value()
        v = constraint.get_value()
        if isinstance(v, UpdateRequest):
            return self.update(key, trace, IncrementalUpdateRequest(argdiffs, v))
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
            value_chm = trace.get_sample()
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
            old_value = trace.get_sample().get_value()

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

    @dispatch
    def update_incremental_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        raise NotImplementedError

    @overload
    def update_incremental(
        self,
        key: PRNGKey,
        trace: Trace,
        request: ConstraintUpdateRequest,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        constraint = request.constraint
        return self.update_incremental_constraint(key, trace, constraint, argdiffs)

    @dispatch
    def update_incremental(
        self,
        key: PRNGKey,
        trace: Trace,
        request: UpdateRequest,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        raise NotImplementedError

    @overload
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: IncrementalUpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        (argdiffs, subrequest) = update_request.argdiffs, update_request.subrequest
        return self.update_incremental(key, trace, subrequest, argdiffs)

    @overload
    def update_importance_value(
        self, key: PRNGKey, trace: Trace, value: None, args: Tuple
    ):
        tr = self.simulate(key, args)
        retdiff = Diff.tree_diff_unknown_change(tr.get_retval())
        return tr, jnp.array(0.0), retdiff, EmptyUpdateRequest()

    @overload
    def update_importance_value(
        self, key: PRNGKey, trace: Trace, mask: Mask, args: Arguments
    ):
        flag, value = mask.flag, mask.value

        def _simulate(key, v):
            score, new_v = self.random_weighted(key, *args)
            w = 0.0
            return (score, w, new_v)

        def _importance(key, v):
            w = self.estimate_logpdf(key, v, *args)
            return (w, w, v)

        score, w, new_v = cond(flag, _importance, _simulate, key, value)
        tr = DistributionTrace(self, args, new_v, score)
        bwd_problem = MaskedUpdateRequest(flag, ProjectUpdateRequest())
        retdiff = Diff.tree_diff_unknown_change(new_v)
        return tr, w, retdiff, bwd_problem

    @overload
    def update_importance_value(
        self, key: PRNGKey, trace: Trace, value: Any, args: Tuple
    ):
        w = self.estimate_logpdf(key, value, *args)
        bwd_problem = ProjectUpdateRequest()
        tr = DistributionTrace(self, args, value, w)
        retdiff = Diff.tree_diff_unknown_change(value)
        return tr, w, retdiff, bwd_problem

    @dispatch
    def update_importance_value(self, key, trace, value, args):
        pass

    @overload
    def update_importance(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: ChoiceMapConstraint,
        args: Tuple,
    ):
        v = constraint.get_value()
        return self.update_importance_value(key, trace, v, args)

    @dispatch
    def update_importance(
        self, key: PRNGKey, trace: Trace, constraint: Constraint, args: Arguments
    ):
        raise NotImplementedError

    @overload
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        request: ImportanceUpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        (args, constraint) = request.args, request.constraint
        return self.update_importance(key, trace, constraint, args)

    @GenerativeFunction.gfi_boundary
    @dispatch
    def update(self, key: PRNGKey, trace: Trace, update_request: UpdateRequest):
        pass

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
        w = self.logpdf(v, *args)
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
        sample: ChoiceMap,
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
    sample: Callable,
    logpdf: Callable,
):
    if not isinstance(sample, Closure):
        sample = Pytree.partial()(sample)

    if not isinstance(logpdf, Closure):
        logpdf = Pytree.partial()(logpdf)

    return ExactDensityFromCallables(sample, logpdf)
