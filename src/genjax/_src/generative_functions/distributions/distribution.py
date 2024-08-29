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
    ChoiceMap,
    Constraint,
    EmptyConstraint,
    EmptyProblem,
    EmptyTrace,
    GenerativeFunction,
    GenericProblem,
    ImportanceProblem,
    Mask,
    MaskedConstraint,
    MaskedProblem,
    ProjectProblem,
    R,
    Retdiff,
    Score,
    Selection,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.generative.choice_map import MaskChm
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import Flag, staged_check
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    Generic,
    PRNGKey,
    typecheck,
)

#####
# DistributionTrace
#####


@Pytree.dataclass
class DistributionTrace(
    Generic[R],
    Trace[R],
):
    gen_fn: GenerativeFunction[R]
    args: tuple[Any, ...]
    value: R
    score: Score

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> R:
        return self.value

    def get_gen_fn(self) -> GenerativeFunction[R]:
        return self.gen_fn

    def get_score(self) -> Score:
        return self.score

    def get_sample(self) -> ChoiceMap:
        return ChoiceMap.value(self.value)


################
# Distribution #
################


class Distribution(Generic[R], GenerativeFunction[R]):
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
    ) -> Score:
        pass

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> Trace[R]:
        (w, v) = self.random_weighted(key, *args)
        tr = DistributionTrace(self, args, v, w)
        return tr

    @typecheck
    def importance_choice_map(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: tuple[Any, ...],
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

                score, w, new_v = jax.lax.cond(
                    flag.f, _importance, _simulate, key, value
                )
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
        args: tuple[Any, ...],
    ) -> tuple[Trace[R], Weight, UpdateProblem]:
        def simulate_branch(key, _, args):
            tr = self.simulate(key, args)
            return (
                tr,
                jnp.array(0.0),
                MaskedProblem(Flag(False), ProjectProblem()),
            )

        def importance_branch(key, constraint, args):
            tr, w = self.importance(key, constraint, args)
            return tr, w, MaskedProblem(Flag(True), ProjectProblem())

        return jax.lax.cond(
            constraint.flag.f,
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
        args: tuple[Any, ...],
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
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
        trace: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
        sample = trace.get_choices()
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
        trace: Trace[R],
        constraint: MaskedConstraint,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
        old_sample = trace.get_choices()

        def update_branch(key, trace, constraint, argdiffs):
            tr, w, rd, _ = self.update(key, trace, GenericProblem(argdiffs, constraint))
            return (
                tr,
                w,
                rd,
                MaskedProblem(Flag(True), old_sample),
            )

        def do_nothing_branch(key, trace, _, argdiffs):
            tr, w, _, _ = self.update(
                key, trace, GenericProblem(argdiffs, EmptyProblem())
            )
            return (
                tr,
                w,
                Diff.tree_diff_unknown_change(tr.get_retval()),
                MaskedProblem(Flag(False), old_sample),
            )

        return jax.lax.cond(
            constraint.flag.f,
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
        trace: Trace[R],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
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
                    EmptyProblem(),
                )

            case MaskedConstraint(flag, problem):
                if staged_check(flag):
                    return self.update(key, trace, GenericProblem(argdiffs, problem))
                else:
                    return self.update_constraint_masked_constraint(
                        key, trace, constraint, argdiffs
                    )

            case ChoiceMap():
                check = constraint.has_value()
                v = constraint.get_value()
                if isinstance(v, UpdateProblem):
                    return self.update(key, trace, GenericProblem(argdiffs, v))
                elif check.concrete_true():
                    fwd = self.estimate_logpdf(key, v, *primals)
                    bwd = trace.get_score()
                    w = fwd - bwd
                    new_tr = DistributionTrace(self, primals, v, fwd)
                    discard = trace.get_choices()
                    retval_diff = Diff.tree_diff_unknown_change(v)
                    return (
                        new_tr,
                        w,
                        retval_diff,
                        discard,
                    )
                elif check.concrete_false():
                    value_chm = trace.get_choices()
                    v = value_chm.get_value()
                    fwd = self.estimate_logpdf(key, v, *primals)
                    bwd = trace.get_score()
                    w = fwd - bwd
                    new_tr = DistributionTrace(self, primals, v, fwd)
                    retval_diff = Diff.tree_diff_no_change(v)
                    return (new_tr, w, retval_diff, EmptyProblem())
                elif isinstance(constraint, MaskChm):
                    # Whether or not the choice map has a value is dynamic...
                    # We must handled with a cond.
                    def _true_branch(key, new_value: R, _):
                        fwd = self.estimate_logpdf(key, new_value, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        return (new_value, w, fwd)

                    def _false_branch(key, _, old_value: R):
                        fwd = self.estimate_logpdf(key, old_value, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        return (old_value, w, fwd)

                    masked_value: Mask[R] = v
                    flag = masked_value.flag
                    new_value: R = masked_value.value
                    old_sample = trace.get_choices()
                    old_value: R = old_sample.get_value()

                    new_value, w, score = jax.lax.cond(
                        flag.f,
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
                        MaskedProblem(flag, old_sample),
                    )
                else:
                    raise Exception(
                        "Only `MaskChm` is currently supported for dynamic flags."
                    )

            case _:
                raise Exception("Unhandled constraint in update.")

    def update_masked(
        self: "Distribution[ArrayLike]",
        key: PRNGKey,
        trace: Trace[ArrayLike],
        flag: Flag,
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[ArrayLike], Weight, Retdiff[ArrayLike], UpdateProblem]:
        old_value = trace.get_retval()
        primals = Diff.tree_primal(argdiffs)
        possible_trace, w, retdiff, bwd_problem = self.update(
            key,
            trace,
            GenericProblem(argdiffs, problem),
        )
        new_value = possible_trace.get_retval()
        w = w * flag
        bwd_problem = MaskedProblem(flag, bwd_problem)
        new_trace = DistributionTrace(
            self,
            primals,
            flag.where(new_value, old_value),
            jnp.asarray(flag.where(possible_trace.get_score(), trace.get_score())),
        )

        return new_trace, w, retdiff, bwd_problem

    def update_project(
        self,
        trace: Trace[R],
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
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
        trace: Trace[R],
        selection: Selection,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
        check = () in selection

        return self.update(
            key,
            trace,
            GenericProblem(
                argdiffs,
                MaskedProblem(Flag(check), ProjectProblem()),
            ),
        )

    @typecheck
    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace[R],
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
        match update_problem:
            case EmptyProblem():
                return self.update_empty(trace, argdiffs)

            case Constraint():
                return self.update_constraint(key, trace, update_problem, argdiffs)

            case MaskedProblem(flag, subproblem):
                # TODO (#1239): see if `MaskedProblem` can handle this update,
                # without infecting all of `Distribution`
                return self.update_masked(key, trace, flag, subproblem, argdiffs)  # pyright: ignore[reportAttributeAccessIssue]

            case ProjectProblem():
                return self.update_project(trace)

            case Selection():
                return self.update_selection_project(
                    key, trace, update_problem, argdiffs
                )

            case ImportanceProblem(constraint) if isinstance(trace, EmptyTrace):
                primals = Diff.tree_primal(argdiffs)
                return self.update_importance(key, constraint, primals)

            case _:
                raise Exception(f"Not implement fwd problem: {update_problem}.")

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace[R],
        update_problem: UpdateProblem,
    ) -> tuple[Trace[R], Weight, Retdiff[R], UpdateProblem]:
        match update_problem:
            case GenericProblem(argdiffs, subproblem):
                return self.update_change_target(key, trace, subproblem, argdiffs)
            case _:
                return self.update_change_target(
                    key, trace, update_problem, Diff.no_change(trace.get_args())
                )

    @typecheck
    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ):
        raise NotImplementedError


################
# ExactDensity #
################


class ExactDensity(Generic[R], Distribution[R]):
    @abstractmethod
    def sample(self, key: PRNGKey, *args) -> R:
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, v: R, *args) -> Score:
        raise NotImplementedError

    def __abstract_call__(self, *args):
        key = jax.random.PRNGKey(0)
        return self.sample(key, *args)

    def handle_kwargs(self) -> GenerativeFunction[R]:
        @Pytree.partial(self)
        def sample_with_kwargs(self: "ExactDensity[R]", key, args, kwargs):
            return self.sample(key, *args, **kwargs)

        @Pytree.partial(self)
        def logpdf_with_kwargs(self: "ExactDensity[R]", v, args, kwargs):
            return self.logpdf(v, *args, **kwargs)

        return ExactDensityFromCallables(
            sample_with_kwargs,
            logpdf_with_kwargs,
        )

    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> tuple[Score, R]:
        """
        Given arguments to the distribution, sample from the distribution, and return the exact log density of the sample, and the sample.
        """
        v = self.sample(key, *args)
        w = self.estimate_logpdf(key, v, *args)
        return (w, v)

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: R,
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
        args: tuple[Any, ...],
    ) -> tuple[Weight, R]:
        key = jax.random.PRNGKey(0)
        v = sample.get_value()
        match v:
            case Mask(flag, value):

                def _check():
                    checkify.check(
                        bool(flag),
                        "Attempted to unmask when a mask flag is False: the masked value is invalid.\n",
                    )

                optional_check(_check)
                w = self.estimate_logpdf(key, value, *args)
                return w, value
            case _:
                w = self.estimate_logpdf(key, v, *args)
                return w, v


@Pytree.dataclass
class ExactDensityFromCallables(Generic[R], ExactDensity[R]):
    sampler: Closure[R]
    logpdf_evaluator: Closure[Score]

    def sample(self, key, *args) -> R:
        return self.sampler(key, *args)

    def logpdf(self, v, *args) -> Score:
        return self.logpdf_evaluator(v, *args)


@typecheck
def exact_density(
    sample: Callable[..., R],
    logpdf: Callable[..., Score],
):
    if not isinstance(sample, Closure):
        sample = Pytree.partial()(sample)

    if not isinstance(logpdf, Closure):
        logpdf = Pytree.partial()(logpdf)

    return ExactDensityFromCallables[R](sample, logpdf)
