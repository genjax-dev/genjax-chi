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
from jax.lax import cond

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    Constraint,
    EmptyConstraint,
    EmptyUpdateSpec,
    GenerativeFunction,
    Mask,
    MaskedConstraint,
    MaskedUpdateSpec,
    RemoveSampleUpdateSpec,
    RemoveSelectionUpdateSpec,
    Retdiff,
    Sample,
    Selection,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import staged_check
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
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
        return ChoiceMap.v(self.value)


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
    ) -> DistributionTrace:
        (w, v) = self.random_weighted(key, *args)
        tr = DistributionTrace(self, args, v, w)
        return tr

    @GenerativeFunction.gfi_boundary
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
                return tr, jnp.array(0.0), EmptyUpdateSpec()

            case Mask(flag, value):

                def _simulate(key, v):
                    tr = self.simulate(key, args)
                    w = 0.0
                    return (tr, w)

                def _importance(key, v):
                    w = self.estimate_logpdf(key, v, *args)
                    tr = DistributionTrace(self, args, v, w)
                    return (tr, w)

                tr, w = cond(flag, _importance, _simulate, key, value)
                bwd_spec = MaskedUpdateSpec(flag, RemoveSampleUpdateSpec())
                return tr, w, bwd_spec

            case _:
                w = self.estimate_logpdf(key, v, *args)
                bwd_spec = RemoveSampleUpdateSpec()
                tr = DistributionTrace(self, args, v, w)
                return tr, w, bwd_spec

    @typecheck
    def importance_masked_constraint(
        self,
        key: PRNGKey,
        constraint: MaskedConstraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        def simulate_branch(key, _, args):
            tr = self.simulate(key, args)
            return tr, jnp.array(0.0), MaskedUpdateSpec(False, RemoveSampleUpdateSpec())

        def importance_branch(key, constraint, args):
            tr, w, _ = self.importance(key, constraint, args)
            return tr, w, MaskedUpdateSpec(True, RemoveSampleUpdateSpec())

        return jax.lax.cond(
            constraint.flag,
            importance_branch,
            simulate_branch,
            key,
            constraint.constraint,
            args,
        )

    @GenerativeFunction.gfi_boundary
    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        match constraint:
            case ChoiceMap():
                return self.importance_choice_map(key, constraint, args)
            case MaskedConstraint(flag, inner_constraint):
                if staged_check(flag):
                    return self.importance(key, inner_constraint, args)
                else:
                    return self.importance_masked_constraint(key, constraint, args)
            case EmptyConstraint():
                tr = self.simulate(key, args)
                return tr, jnp.array(0.0), EmptyUpdateSpec()
            case _:
                raise Exception("Unhandled type.")

    def update_empty(
        self,
        trace: DistributionTrace,
        argdiffs: Argdiffs,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        sample = trace.get_sample()
        primals = Diff.tree_primal(argdiffs)
        new_score, _ = self.assess(sample, primals)
        new_trace = DistributionTrace(self, primals, sample.get_value(), new_score)
        return (
            new_trace,
            new_score - trace.get_score(),
            Diff.tree_diff_no_change(trace.get_retval()),
            EmptyUpdateSpec(),
        )

    def update_constraint_masked_constraint(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        constraint: MaskedConstraint,
        argdiffs: Argdiffs,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        old_sample = trace.get_sample()
        primals = Diff.tree_primal(argdiffs)

        def update_branch(key, trace, constraint, argdiffs):
            tr, w, rd, _ = self.update(key, trace, constraint, argdiffs)
            return (
                tr,
                w,
                rd,
                MaskedUpdateSpec(True, old_sample),
            )

        def do_nothing_branch(key, trace, constraint, argdiffs):
            tr, w, _, _ = self.update(key, trace, EmptyUpdateSpec(), argdiffs)
            return (
                tr,
                w,
                Diff.tree_diff_unknown_change(tr.get_retval()),
                MaskedUpdateSpec(False, old_sample),
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
        trace: DistributionTrace,
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
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
                    EmptyUpdateSpec(),
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
                if isinstance(v, UpdateSpec):
                    return self.update(key, trace, v, argdiffs)
                elif static_check_is_concrete(check) and check:
                    fwd = self.estimate_logpdf(key, v, *primals)
                    bwd = trace.get_score()
                    w = fwd - bwd
                    new_tr = DistributionTrace(self, primals, v, fwd)
                    discard = trace.get_sample()
                    retval_diff = Diff.tree_diff_unknown_change(v)
                    return (new_tr, w, retval_diff, discard)
                elif static_check_is_concrete(check):
                    value_chm = trace.get_sample()
                    v = value_chm.get_value()
                    fwd = self.estimate_logpdf(key, v, *primals)
                    bwd = trace.get_score()
                    w = fwd - bwd
                    new_tr = DistributionTrace(self, primals, v, fwd)
                    retval_diff = Diff.tree_diff_no_change(v)
                    return (new_tr, w, retval_diff, EmptyUpdateSpec())
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
                        MaskedUpdateSpec(flag, old_value),
                    )

            case _:
                raise Exception("Unhandled constraint in update.")

    # TODO: check math.
    def update_remove_sample(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        argdiffs: Argdiffs,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        gen_fn = trace.get_gen_fn()
        original = trace.get_score()
        removed_value = trace.get_retval()
        primals = Diff.tree_primal(argdiffs)
        new_tr = gen_fn.simulate(key, primals)
        retdiff = Diff.tree_diff_unknown_change(new_tr.get_retval())
        return (
            new_tr,
            -original,
            retdiff,
            ChoiceMap.v(removed_value),
        )

    def update_masked_spec(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        update_spec: MaskedUpdateSpec,
        argdiffs: Argdiffs,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        old_value = trace.get_retval()
        primals = Diff.tree_primal(argdiffs)
        match update_spec:
            case MaskedUpdateSpec(flag, spec):
                possible_trace, w, retdiff, bwd_spec = self.update(
                    key, trace, spec, argdiffs
                )
                new_value = possible_trace.get_retval()
                w = w * flag
                bwd_spec = MaskedUpdateSpec(flag, bwd_spec)
                new_trace = DistributionTrace(
                    self,
                    primals,
                    jax.lax.select(flag, new_value, old_value),
                    jax.lax.select(flag, possible_trace.get_score(), trace.get_score()),
                )

                return new_trace, w, retdiff, bwd_spec

    def update_remove_selection(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        selection: Selection,
        argdiffs: Argdiffs,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        check, _ = selection.has_addr(())
        return self.update(
            key,
            trace,
            MaskedUpdateSpec.maybe(check, RemoveSampleUpdateSpec()),
            argdiffs,
        )

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case EmptyUpdateSpec():
                return self.update_empty(trace, argdiffs)

            case Constraint():
                return self.update_constraint(key, trace, update_spec, argdiffs)

            case MaskedUpdateSpec(flag, spec):
                return self.update_masked_spec(key, trace, update_spec, argdiffs)

            case RemoveSampleUpdateSpec():
                return self.update_remove_sample(key, trace, argdiffs)

            case RemoveSelectionUpdateSpec(selection):
                return self.update_remove_selection(key, trace, selection, argdiffs)

            case _:
                raise Exception(f"Not implement fwd spec: {update_spec}.")

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
        sample: Sample,
        args: Tuple,
    ):
        key = jax.random.PRNGKey(0)
        tr, w, _ = self.importance(key, sample, args)
        return w, tr.get_retval()


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
