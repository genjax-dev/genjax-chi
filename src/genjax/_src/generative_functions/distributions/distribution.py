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

import abc

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
    EmptyProblem,
    EmptyTrace,
    GenerativeFunction,
    ImportanceProblem,
    Mask,
    MaskedConstraint,
    MaskedProblem,
    ProjectProblem,
    Retdiff,
    Sample,
    Selection,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import staged_check
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Bool,
    BoolArray,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
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


class Distribution(GenerativeFunction):
    @abc.abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Tuple[FloatArray, Any]:
        pass

    @abc.abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *args,
    ) -> FloatArray:
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
    def importance_choice_map(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ):
        v = chm.get_value()
        match v:
            case None:
                tr = self.simulate(key, args)
                return tr, jnp.array(0.0), EmptyProblem()

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
                bwd_problem = MaskedProblem(flag, ProjectProblem())
                return tr, w, bwd_problem

            case _:
                w = self.estimate_logpdf(key, v, *args)
                bwd_problem = ProjectProblem()
                tr = DistributionTrace(self, args, v, w)
                return tr, w, bwd_problem

    @typecheck
    def importance_masked_constraint(
        self,
        key: PRNGKey,
        constraint: MaskedConstraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, UpdateProblem]:
        def simulate_branch(key, _, args):
            tr = self.simulate(key, args)
            return (
                tr,
                jnp.array(0.0),
                MaskedProblem(False, ProjectProblem()),
            )

        def importance_branch(key, constraint, args):
            tr, w = self.importance(key, constraint, args)
            return tr, w, MaskedProblem(True, ProjectProblem())

        return jax.lax.cond(
            constraint.flag,
            importance_branch,
            simulate_branch,
            key,
            constraint.constraint,
            args,
        )

    @typecheck
    def update_importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match constraint:
            case ChoiceMap():
                tr, w, bwd_problem = self.importance_choice_map(key, constraint, args)
            case MaskedConstraint(flag, inner_constraint):
                if staged_check(flag):
                    return self.update_importance(key, inner_constraint, args)
                else:
                    tr, w, bwd_problem = self.importance_masked_constraint(
                        key, constraint, args
                    )
            case EmptyConstraint():
                tr = self.simulate(key, args)
                w = jnp.array(0.0)
                bwd_problem = EmptyProblem()
            case _:
                raise Exception("Unhandled type.")
        return tr, w, Diff.unknown_change(tr.get_retval()), bwd_problem

    def update_empty(
        self,
        trace: Trace,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        sample = trace.get_sample()
        primals = Diff.tree_primal(argdiffs)
        new_score, _ = self.assess(sample, primals)
        new_trace = DistributionTrace(self, primals, sample.get_value(), new_score)
        return (
            new_trace,
            new_score - trace.get_score(),
            Diff.tree_diff_no_change(trace.get_retval()),
            EmptyProblem(),
        )

    def update_constraint_masked_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: MaskedConstraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        old_sample = trace.get_sample()

        def update_branch(key, trace, constraint, argdiffs):
            tr, w, rd, _ = self.update(key, trace, constraint, argdiffs)
            return (
                tr,
                w,
                rd,
                MaskedProblem(True, old_sample),
            )

        def do_nothing_branch(key, trace, constraint, argdiffs):
            tr, w, _, _ = self.update(key, trace, EmptyProblem(), argdiffs)
            return (
                tr,
                w,
                Diff.tree_diff_unknown_change(tr.get_retval()),
                MaskedProblem(False, old_sample),
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

    def update_constraint(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        primals = Diff.tree_primal(argdiffs)
        match constraint:
            case EmptyConstraint():
                old_sample = trace.get_sample()
                old_retval = trace.get_retval()
                new_score, _ = self.assess(old_sample, primals)
                new_trace = DistributionTrace(
                    self, primals, old_sample.get_value(), new_score
                )
                return (
                    new_trace,
                    new_score - trace.get_score(),
                    Diff.tree_diff_no_change(old_retval),
                    EmptyProblem(),
                )

            case MaskedConstraint(flag, spec):
                if staged_check(flag):
                    return self.update(key, trace, spec, argdiffs)
                else:
                    return self.update_constraint_masked_constraint(
                        key, trace, constraint, argdiffs
                    )

            case ChoiceMap():
                check = constraint.has_value()
                v = constraint.get_value()
                if isinstance(v, UpdateProblem):
                    return self.update(key, trace, v, argdiffs)
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
                    return (new_tr, w, retval_diff, EmptyProblem())
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
                        MaskedProblem(flag, old_value),
                    )

            case _:
                raise Exception("Unhandled constraint in update.")

    def update_masked(
        self,
        key: PRNGKey,
        trace: Trace,
        flag: Bool | BoolArray,
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        old_value = trace.get_retval()
        primals = Diff.tree_primal(argdiffs)
        possible_trace, w, retdiff, bwd_problem = self.update(
            key, trace, problem, argdiffs
        )
        new_value = possible_trace.get_retval()
        w = w * flag
        bwd_problem = MaskedProblem(flag, bwd_problem)
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
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
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
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        check = () in selection

        return self.update(
            key,
            trace,
            MaskedProblem.maybe(check, ProjectProblem()),
            argdiffs,
        )

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case EmptyProblem():
                return self.update_empty(trace, argdiffs)

            case Constraint():
                return self.update_constraint(key, trace, update_problem, argdiffs)

            case MaskedProblem(flag, subproblem):
                return self.update_masked(key, trace, flag, subproblem, argdiffs)

            case ProjectProblem():
                return self.update_project(trace)

            case Selection():
                return self.update_selection_project(
                    key, trace, update_problem, argdiffs
                )

            case ImportanceProblem(constraint) if isinstance(trace, EmptyTrace):
                return self.update_importance(
                    key, constraint, Diff.tree_primal(argdiffs)
                )

            case _:
                raise Exception(f"Not implement fwd problem: {update_problem}.")

    @GenerativeFunction.gfi_boundary
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


@Pytree.dataclass
class ExactDensity(Distribution):
    sampler: Closure
    logpdf_evaluator: Closure

    def __abstract_call__(self, *args):
        key = jax.random.PRNGKey(0)
        return self.sampler(key, *args)

    def handle_kwargs(self) -> GenerativeFunction:
        @Pytree.partial(self)
        def sampler_with_kwargs(self, key, args, kwargs):
            return self.sampler(key, *args, **kwargs)

        @Pytree.partial(self)
        def logpdf_with_kwargs(self, v, args, kwargs):
            return self.logpdf_evaluator(v, *args, **kwargs)

        return ExactDensity(
            sampler_with_kwargs,
            logpdf_with_kwargs,
        )

    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Tuple[FloatArray, Any]:
        v = self.sampler(key, *args)
        w = self.logpdf_evaluator(v, *args)
        return (w, v)

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *args,
    ) -> FloatArray:
        w = self.logpdf_evaluator(v, *args)
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


@typecheck
def exact_density(
    sampler: Callable,
    logpdf: Callable,
):
    if not isinstance(sampler, Closure):
        sampler = Pytree.partial()(sampler)

    if not isinstance(logpdf, Closure):
        logpdf = Pytree.partial()(logpdf)

    return ExactDensity(sampler, logpdf)
