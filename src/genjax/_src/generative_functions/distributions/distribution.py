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
    ChoiceMap,
    Constraint,
    EmptyUpdateSpec,
    GenerativeFunction,
    Mask,
    MaskUpdateSpec,
    RemoveSampleUpdateSpec,
    RemoveSelectionUpdateSpec,
    Retdiff,
    Selection,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
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

    def get_retval(self):
        return self.value

    def get_gen_fn(self):
        return self.gen_fn

    def get_score(self):
        return self.score

    def get_sample(self):
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
        check = chm.has_value()
        if static_check_is_concrete(check) and check:
            v = chm.get_value()
            w = self.estimate_logpdf(key, v, *args)
            score = w
            bwd_spec = RemoveSampleUpdateSpec(v)
            return (DistributionTrace(self, args, v, score), w, bwd_spec)
        elif static_check_is_concrete(check):
            score, v = self.random_weighted(key, *args)
            return (
                DistributionTrace(self, args, v, score),
                jnp.array(0.0),
                EmptyUpdateSpec(),
            )
        else:
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
                    bwd_spec = MaskUpdateSpec(flag, RemoveSampleUpdateSpec())
                    return tr, w, bwd_spec

                case _:
                    raise Exception("Unhandled type.")

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
            case _:
                raise Exception("Unhandled type.")

    def update_empty(
        self,
        trace: DistributionTrace,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        return (
            trace,
            jnp.array(0.0),
            Diff.tree_diff_no_change(trace.get_retval()),
            EmptyUpdateSpec(),
        )

    def update_constraint(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        constraint: Constraint,
        argdiffs: Tuple,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        check = constraint.has_value()
        primals = Diff.tree_primals(argdiffs)
        if static_check_is_concrete(check) and check:
            v = constraint.get_value()
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
            raise NotImplementedError

    # TODO: check math.
    def update_remove_sample(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        argdiffs: Tuple,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        gen_fn = trace.get_gen_fn()
        original = trace.get_score()
        removed_value = trace.get_retval()
        new_tr = gen_fn.simulate(key)
        retdiff = Diff.tree_diff_unknown_change(new_tr.get_retval())
        return new_tr, -original, retdiff, ChoiceMap.v(removed_value)

    def update_mask_spec(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        update_spec: MaskUpdateSpec,
        argdiffs: Tuple,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        old_value = trace.get_retval()
        primals = Diff.tree_primals(argdiffs)
        match update_spec:
            case MaskUpdateSpec(flag, spec):
                possible_trace, w, retdiff, bwd_spec = self.update(
                    key, trace, spec, argdiffs
                )
                new_value = possible_trace.get_retval()
                w = w * flag
                bwd_spec = MaskUpdateSpec(flag, bwd_spec)
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
        argdiffs: Tuple,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        check, _ = selection.has_addr(())
        return self.update(
            key, trace, MaskUpdateSpec.maybe(check, RemoveSampleUpdateSpec()), argdiffs
        )

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case EmptyUpdateSpec():
                return self.update_empty(trace)

            case Constraint():
                return self.update_constraint(key, trace, update_spec, argdiffs)

            case MaskUpdateSpec(flag, spec):
                return self.update_mask_spec(key, trace, update_spec, argdiffs)

            case RemoveSampleUpdateSpec():
                return self.update_remove_sample(key, trace, argdiffs)

            case RemoveSelectionUpdateSpec(selection):
                return self.update_remove_selection(key, trace, selection, argdiffs)

            case _:
                raise Exception(f"Not implement fwd spec: {update_spec}.")


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


@typecheck
def exact_density(
    sampler: Callable,
    logpdf: Callable,
):
    return ExactDensity(sampler, logpdf)
