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
from jax.experimental import checkify
from jax.lax import cond

from genjax._src.checkify import optional_check
from genjax._src.core.generative import (
    Argdiffs,
    Arguments,
    ChangeTargetWithConstraintRequest,
    ChoiceMap,
    Constraint,
    EmptyConstraint,
    EmptyRequest,
    EmptyTrace,
    GenerativeFunction,
    ImportanceRequest,
    Mask,
    MaskedConstraint,
    MaskedRequest,
    ProjectRequest,
    Retdiff,
    Retval,
    Sample,
    Score,
    Selection,
    Trace,
    UnhandledConstraint,
    UnhandledRequest,
    UpdateRequest,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Bool,
    BoolArray,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
    typecheck,
)

#####################
# DistributionTrace #
#####################


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


class Distribution(GenerativeFunction):
    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Tuple[Score, Retval]:
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
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

    @typecheck
    def update_importance_choice_map(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        v = chm.get_value()
        match v:
            case None:
                tr = self.simulate(key, args)
                retdiff = Diff.unknown_change(tr.get_retval())
                return tr, jnp.array(0.0), retdiff, EmptyRequest()

            case Mask(flag, value):

                def _simulate(key, v):
                    score, new_v = self.random_weighted(key, *args)
                    w = 0.0
                    return (score, w, new_v)

                def _importance(key, v):
                    w = self.estimate_logpdf(key, v, *args)
                    return (w, w, v)

                score, w, new_v = cond(flag, _importance, _simulate, key, value)
                tr = DistributionTrace(self, args, new_v, score)
                bwd_problem = MaskedRequest(flag, ProjectRequest())
                retdiff = Diff.unknown_change(tr.get_retval())
                return tr, w, retdiff, bwd_problem

            case _:
                w = self.estimate_logpdf(key, v, *args)
                bwd_problem = ProjectRequest()
                tr = DistributionTrace(self, args, v, w)
                retdiff = Diff.unknown_change(tr.get_retval())
                return tr, w, retdiff, bwd_problem

    @typecheck
    def update_importance_masked_constraint(
        self,
        key: PRNGKey,
        flag: Bool | BoolArray,
        constraint: MaskedConstraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        def simulate_branch(key, _, args):
            tr = self.simulate(key, args)
            return (
                tr,
                jnp.array(0.0),
                MaskedRequest(False, ProjectRequest()),
            )

        def importance_branch(key, constraint, args):
            tr, w = self.importance(key, constraint, args)
            return tr, w, MaskedRequest(True, ProjectRequest())

        tr, w, bwd_request = jax.lax.cond(
            flag,
            importance_branch,
            simulate_branch,
            key,
            constraint,
            args,
        )
        retdiff = Diff.unknown_change(tr.get_retval())
        return tr, w, retdiff, bwd_request

    @typecheck
    def update_importance_empty_constraint(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        tr = self.simulate(key, args)
        w = jnp.array(0.0)
        bwd_problem = EmptyRequest()
        retdiff = Diff.unknown_change(tr.get_retval())
        return tr, w, retdiff, bwd_problem

    @typecheck
    def update_importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        match constraint:
            case ChoiceMap():
                return self.update_importance_choice_map(
                    key,
                    constraint,
                    args,
                )
            case MaskedConstraint(flag, constraint):
                return self.update_importance_masked_constraint(
                    key,
                    flag,
                    constraint,
                    args,
                )
            case EmptyConstraint():
                return self.update_importance_empty_constraint(
                    key,
                    args,
                )
            case _:
                raise UnhandledConstraint(constraint)

    @typecheck
    def update_empty(
        self,
        trace: Trace,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        return (
            trace,
            0.0,
            Diff.tree_diff_no_change(trace.get_retval()),
            EmptyRequest(),
        )

    @typecheck
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

    @typecheck
    def update_selection(
        self,
        key: PRNGKey,
        trace: Trace,
        selection: Selection,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        check = () in selection

        return self.update(
            key,
            trace,
            MaskedRequest.maybe_empty(check, ProjectRequest()),
        )

    @typecheck
    def update_masked(
        self,
        key: PRNGKey,
        trace: Trace,
        flag: Bool | BoolArray,
        problem: UpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        possible_trace, w, retdiff, bwd_problem = self.update(
            key,
            trace,
            problem,
        )
        old_value = trace.get_retval()
        new_value = possible_trace.get_retval()
        w = w * flag
        bwd_problem = MaskedRequest(flag, bwd_problem)
        new_trace = DistributionTrace(
            self,
            jtu.tree_map(
                lambda v1, v2: jax.lax.select(flag, v1, v2),
                trace.get_args(),
                possible_trace.get_args(),
            ),
            jax.lax.select(flag, new_value, old_value),
            jax.lax.select(flag, possible_trace.get_score(), trace.get_score()),
        )

        return new_trace, w, retdiff, bwd_problem

    @typecheck
    def update_ctc_empty(
        self,
        trace: Trace,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        pass

    @typecheck
    def update_ctc_masked_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        flag: Bool | BoolArray,
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        pass

    @typecheck
    def update_ctc_choice_map(
        self,
        key: PRNGKey,
        trace: Trace,
        chm: ChoiceMap,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        pass

    @typecheck
    def update_change_target_with_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        match constraint:
            case EmptyConstraint():
                return self.update_ctc_empty(
                    trace,
                    argdiffs,
                )

            case MaskedConstraint(flag, constraint):
                return self.update_ctc_masked_constraint(
                    key,
                    trace,
                    flag,
                    constraint,
                    argdiffs,
                )

            case ChoiceMap():
                return self.update_ctc_choice_map(
                    key,
                    trace,
                    constraint,
                    argdiffs,
                )

            case _:
                raise UnhandledConstraint(constraint)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: UpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        match update_request:
            case EmptyRequest():
                return self.update_empty(
                    trace,
                )

            case MaskedRequest(flag, subrequest):
                return self.update_masked(
                    key,
                    trace,
                    flag,
                    subrequest,
                )

            case ChangeTargetWithConstraintRequest(argdiffs, constraint):
                return self.update_change_target_with_constraint(
                    key,
                    trace,
                    constraint,
                    argdiffs,
                )

            case ImportanceRequest(args, constraint) if isinstance(trace, EmptyTrace):
                return self.update_importance(
                    key,
                    constraint,
                    args,
                )

            case Selection():
                return self.update_selection(
                    key,
                    trace,
                    update_request,
                )

            case ProjectRequest():
                return self.update_project(
                    trace,
                )

            case _:
                raise UnhandledRequest(update_request)

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
