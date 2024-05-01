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
import jax.tree_util as jtu
from jax.lax import cond

from genjax._src.core.generative import (
    ChangeTargetUpdateSpec,
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
from genjax._src.core.pytree import Pytree
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
    value: Any
    score: FloatArray

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.value

    def get_score(self):
        return self.score

    def get_sample(self):
        return ChoiceMap.v(self.value)

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        check = selection[...]
        return check * self.get_score()

    def get_value(self):
        return self.value


#####
# Distribution
#####


class Distribution(GenerativeFunction):
    @abc.abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
    ) -> Tuple[FloatArray, Any]:
        pass

    @abc.abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
    ) -> FloatArray:
        pass

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
    ) -> DistributionTrace:
        (w, v) = self.random_weighted(key)
        tr = DistributionTrace(self, v, w)
        return tr

    @typecheck
    def importance_choice_map(self, key: PRNGKey, chm: ChoiceMap):
        check = chm.has_value()
        if static_check_is_concrete(check) and check:
            v = chm.get_value()
            w = self.estimate_logpdf(key, v)
            score = w
            bwd_spec = RemoveSampleUpdateSpec(v)
            return (DistributionTrace(self, v, score), w, bwd_spec)
        elif static_check_is_concrete(check):
            score, v = self.random_weighted(key)
            return (
                DistributionTrace(self, v, score),
                jnp.array(0.0),
                EmptyUpdateSpec(),
            )
        else:
            v = chm.get_value()
            match v:
                case Mask(flag, value):

                    def _simulate(key, v):
                        tr = self.simulate(key)
                        w = 0.0
                        return (tr, w)

                    def _importance(key, v):
                        w = self.estimate_logpdf(key, v)
                        tr = DistributionTrace(self, v, w)
                        return (tr, w)

                    tr, w = cond(flag, _importance, _simulate, key, value)
                    bwd_spec = MaskUpdateSpec(flag, RemoveSampleUpdateSpec())
                    return tr, w, bwd_spec

                case _:
                    raise Exception("Unhandled type.")

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        match constraint:
            case ChoiceMap():
                return self.importance_choice_map(key, constraint)
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
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        check = constraint.has_value()
        if static_check_is_concrete(check) and check:
            v = constraint.get_value()
            fwd = self.estimate_logpdf(key, v)
            bwd = trace.get_score()
            w = fwd - bwd
            new_tr = DistributionTrace(self, v, fwd)
            discard = trace.get_sample()
            retval_diff = Diff.tree_diff_unknown_change(v)
            return (new_tr, w, retval_diff, discard)
        elif static_check_is_concrete(check):
            value_chm = trace.get_sample()
            v = value_chm.get_value()
            fwd = self.estimate_logpdf(key, v, *args)
            bwd = trace.get_score()
            w = fwd - bwd
            new_tr = DistributionTrace(self, v, fwd)
            retval_diff = Diff.tree_diff_no_change(v)
            return (new_tr, w, retval_diff, EmptyUpdateSpec())
        else:
            raise NotImplementedError

    def update_remove_sample(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        gen_fn = trace.get_gen_fn()
        original = trace.get_score()
        removed_value = trace.get_retval()
        new_tr = gen_fn.simulate(key)
        retdiff = Diff.tree_diff_unknown_change(new_tr.get_retval())
        return new_tr, -original, retdiff, ChoiceMap.v(removed_value)

    def update_remove_selection(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        selection: Selection,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        check = () in selection
        return self.update(
            key, trace, MaskUpdateSpec.maybe(check, RemoveSampleUpdateSpec())
        )

    def update_change_target(
        self,
        key: PRNGKey,
        trace: DistributionTrace,
        argdiffs: Tuple,
        inner_spec: UpdateSpec,
    ) -> Tuple[DistributionTrace, Weight, Retdiff, UpdateSpec]:
        Diff.static_check_tree_diff(argdiffs)
        args = Diff.tree_primal(argdiffs)
        dist_def = jtu.tree_structure(self)
        new_dist = jtu.tree_unflatten(dist_def, args)
        return new_dist.update(key, trace, inner_spec)

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case EmptyUpdateSpec():
                return self.update_empty(trace)

            case Constraint():
                return self.update_constraint(key, trace, spec)

            case ChangeTargetUpdateSpec(argdiffs, spec):
                return self.update_change_target(key, trace, argdiffs, spec)

            case RemoveSampleUpdateSpec():
                return self.update_remove_sample(key, trace)

            case RemoveSelectionUpdateSpec(selection):
                return self.update_remove_selection(key, trace, selection)

            case _:
                raise Exception(f"Not implement fwd spec: {update_spec}.")


#####
# ExactDensity
#####


@Pytree.dataclass
class ExactDensity(Distribution):
    args: Tuple
    sampler: Callable = Pytree.static()
    logpdf_evaluator: Callable = Pytree.static()
    kwargs: dict = Pytree.field(default_factory=dict)

    def __abstract_call__(self):
        key = jax.random.PRNGKey(0)
        return self.sampler(key, *self.args, **self.kwargs)

    def random_weighted(
        self,
        key: PRNGKey,
    ) -> Tuple[FloatArray, Any]:
        v = self.sampler(key, *self.args, **self.kwargs)
        w = self.logpdf_evaluator(v, *self.args, **self.kwargs)
        return (w, v)

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
    ) -> FloatArray:
        w = self.logpdf_evaluator(v, *self.args, **self.kwargs)
        if w.shape:
            return jnp.sum(w)
        else:
            return w


@typecheck
def exact_density(sampler: Callable, logpdf: Callable):
    def inner(*args, **kwargs):
        return ExactDensity(args, sampler, logpdf, kwargs)

    return inner
