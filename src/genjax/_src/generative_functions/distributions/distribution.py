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
    ChoiceMapConstraint,
    Constraint,
    EditRequest,
    EmptyConstraint,
    GenerativeFunction,
    IncrementalGenericRequest,
    Mask,
    Projection,
    R,
    Retdiff,
    Score,
    Selection,
    Trace,
    Weight,
)
from genjax._src.core.generative.choice_map import Filtered
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import FlagOp
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    PRNGKey,
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
        return self.get_choices()

    def get_choices(self) -> ChoiceMap:
        return ChoiceMap.choice(self.value)


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

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> Trace[R]:
        (w, v) = self.random_weighted(key, *args)
        tr = DistributionTrace(self, args, v, w)
        return tr

    def generate_choice_map(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Trace[R], Weight]:
        v = chm.get_value()
        match v:
            case None:
                tr = self.simulate(key, args)
                return tr, jnp.array(0.0)

            case Mask(flag, value):

                def _simulate(key, v):
                    score, new_v = self.random_weighted(key, *args)
                    w = 0.0
                    return (score, w, new_v)

                def _importance(key, v):
                    w = self.estimate_logpdf(key, v, *args)
                    return (w, w, v)

                score, w, new_v = jax.lax.cond(flag, _importance, _simulate, key, value)
                tr = DistributionTrace(self, args, new_v, score)
                return tr, w

            case _:
                w = self.estimate_logpdf(key, v, *args)
                tr = DistributionTrace(self, args, v, w)
                return tr, w

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[Trace[R], Weight]:
        match constraint:
            case ChoiceMapConstraint():
                tr, w = self.generate_choice_map(key, constraint, args)
            case EmptyConstraint():
                tr = self.simulate(key, args)
                w = jnp.array(0.0)
            case _:
                raise Exception("Unhandled type.")
        return tr, w

    def edit_empty(
        self,
        trace: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], IncrementalGenericRequest]:
        sample = trace.get_choices()
        primals = Diff.tree_primal(argdiffs)
        new_score, _ = self.assess(sample, primals)
        new_trace = DistributionTrace(self, primals, sample.get_value(), new_score)
        return (
            new_trace,
            new_score - trace.get_score(),
            Diff.tree_diff_no_change(trace.get_retval()),
            IncrementalGenericRequest(
                ChoiceMapConstraint(ChoiceMap.empty()),
            ),
        )

    def edit_constraint(
        self,
        key: PRNGKey,
        trace: Trace[R],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], IncrementalGenericRequest]:
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
                    IncrementalGenericRequest(
                        ChoiceMapConstraint(ChoiceMap.empty()),
                    ),
                )

            case ChoiceMapConstraint():
                check = constraint.has_value()
                v = constraint.get_value()
                if FlagOp.concrete_true(check):
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
                        IncrementalGenericRequest(
                            ChoiceMapConstraint(discard),
                        ),
                    )
                elif FlagOp.concrete_false(check):
                    value_chm = trace.get_choices()
                    v = value_chm.get_value()
                    fwd = self.estimate_logpdf(key, v, *primals)
                    bwd = trace.get_score()
                    w = fwd - bwd
                    new_tr = DistributionTrace(self, primals, v, fwd)
                    retval_diff = Diff.tree_diff_no_change(v)
                    return (
                        new_tr,
                        w,
                        retval_diff,
                        IncrementalGenericRequest(
                            ChoiceMapConstraint(ChoiceMap.empty()),
                        ),
                    )

                elif isinstance(constraint.choice_map, Filtered):
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
                    flag = masked_value.primal_flag()
                    new_value: R = masked_value.value
                    old_choices = trace.get_choices()
                    old_value: R = old_choices.get_value()

                    new_value, w, score = FlagOp.cond(
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
                        IncrementalGenericRequest(
                            ChoiceMapConstraint(old_choices.mask(flag)),
                        ),
                    )
                else:
                    raise Exception(
                        "Only `choice_map.Filtered` is currently supported for dynamic flags."
                    )

            case _:
                raise Exception("Unhandled constraint in edit.")

    def project(
        self,
        key: PRNGKey,
        trace: Trace[R],
        projection: Projection[Any],
    ) -> Weight:
        assert isinstance(projection, Selection)
        return jnp.where(
            projection.check(),
            trace.get_score(),
            jnp.array(0.0),
        )

    def edit_incremental_generic_request(
        self,
        key: PRNGKey,
        trace: Trace[R],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], IncrementalGenericRequest]:
        match constraint:
            case EmptyConstraint():
                return self.edit_empty(trace, argdiffs)

            case ChoiceMapConstraint():
                return self.edit_constraint(key, trace, constraint, argdiffs)

            case _:
                raise Exception(f"Not implement fwd problem: {constraint}.")

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        assert isinstance(edit_request, IncrementalGenericRequest)
        constraint = edit_request.constraint
        return self.edit_incremental_generic_request(
            key,
            trace,
            constraint,
            argdiffs,
        )

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


def exact_density(
    sample: Callable[..., R],
    logpdf: Callable[..., Score],
):
    if not isinstance(sample, Closure):
        sample = Pytree.partial()(sample)

    if not isinstance(logpdf, Closure):
        logpdf = Pytree.partial()(logpdf)

    return ExactDensityFromCallables[R](sample, logpdf)
