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
from tensorflow_probability.substrates import jax as tfp

from genjax._src.checkify import optional_check
from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    Constraint,
    EditRequest,
    EmptyConstraint,
    GenerativeFunction,
    Mask,
    NotSupportedEditRequest,
    Projection,
    R,
    Regenerate,
    Retdiff,
    Score,
    Selection,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.generative.choice_map import ChoiceMapConstraint
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import FlagOp, to_shape_fn
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    PRNGKey,
)

tfd = tfp.distributions

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

            case Mask(value, flag):

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
            case ChoiceMapConstraint(chm):
                tr, w = self.generate_choice_map(key, chm, args)
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
    ) -> tuple[Trace[R], Weight, Retdiff[R], Update]:
        sample = trace.get_choices()
        primals = Diff.tree_primal(argdiffs)
        new_score, _ = self.assess(sample, primals)
        new_trace = DistributionTrace(self, primals, sample.get_value(), new_score)
        return (
            new_trace,
            new_score - trace.get_score(),
            Diff.no_change(trace.get_retval()),
            Update(ChoiceMap.empty()),
        )

    def edit_update_with_constraint(
        self,
        key: PRNGKey,
        trace: Trace[R],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], Update]:
        primals = Diff.tree_primal(argdiffs)
        match constraint:
            case ChoiceMapConstraint(chm):
                match chm.get_value():
                    case Mask() as masked_value:

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
                            Diff.unknown_change(new_value),
                            Update(
                                old_choices.mask(flag),
                            ),
                        )
                    case None:
                        value_chm = trace.get_choices()
                        v = value_chm.get_value()
                        fwd = self.estimate_logpdf(key, v, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        new_tr = DistributionTrace(self, primals, v, fwd)
                        retval_diff = Diff.no_change(v)
                        return (new_tr, w, retval_diff, Update(ChoiceMap.empty()))

                    case v:
                        fwd = self.estimate_logpdf(key, v, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        new_tr = DistributionTrace(self, primals, v, fwd)
                        discard = trace.get_choices()
                        retval_diff = Diff.unknown_change(v)
                        return (new_tr, w, retval_diff, Update(discard))
            case _:
                raise Exception(f"Unhandled constraint in edit: {type(constraint)}.")

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

    def edit_regenerate(
        self,
        key: PRNGKey,
        trace: Trace[R],
        selection: Selection,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        check = () in selection
        if FlagOp.concrete_true(check):
            primals = Diff.tree_primal(argdiffs)
            w, new_v = self.random_weighted(key, *primals)
            incremental_w = w - trace.get_score()
            old_v = trace.get_retval()
            new_trace = DistributionTrace(self, primals, new_v, w)
            return (
                new_trace,
                incremental_w,
                Diff.unknown_change(new_v),
                Update(ChoiceMap.choice(old_v)),
            )
        elif FlagOp.concrete_false(check):
            if Diff.static_check_no_change(argdiffs):
                return (
                    trace,
                    jnp.array(0.0),
                    Diff.no_change(trace.get_retval()),
                    Update(ChoiceMap.empty()),
                )
            else:
                chm = trace.get_choices()
                primals = Diff.tree_primal(argdiffs)
                new_score, _ = self.assess(chm, primals)
                new_trace = DistributionTrace(self, primals, chm.get_value(), new_score)
                return (
                    new_trace,
                    new_score - trace.get_score(),
                    Diff.no_change(trace.get_retval()),
                    Update(
                        ChoiceMap.empty(),
                    ),
                )
        else:
            raise NotImplementedError

    def edit_update(
        self,
        key: PRNGKey,
        trace: Trace[R],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], Update]:
        match constraint:
            case EmptyConstraint():
                return self.edit_empty(trace, argdiffs)

            case ChoiceMapConstraint():
                return self.edit_update_with_constraint(
                    key, trace, constraint, argdiffs
                )

            case _:
                raise Exception(f"Not implement fwd problem: {constraint}.")

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        match edit_request:
            case Update(chm):
                return self.edit_update(
                    key,
                    trace,
                    ChoiceMapConstraint(chm),
                    argdiffs,
                )
            case Regenerate(selection):
                return self.edit_regenerate(
                    key,
                    trace,
                    selection,
                    argdiffs,
                )

            case _:
                raise NotSupportedEditRequest(edit_request)

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ):
        raise NotImplementedError


################
# ExactDensity #
################

_fake_key = jnp.array([0, 0], dtype=jnp.uint32)


class ExactDensity(Generic[R], Distribution[R]):
    @abstractmethod
    def sample(self, key: PRNGKey, *args) -> R:
        pass

    @abstractmethod
    def logpdf(self, v: R, *args) -> Score:
        pass

    def __abstract_call__(self, *args):
        return to_shape_fn(self.sample, jnp.zeros)(_fake_key, *args)

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
        key = jax.random.key(0)
        v = sample.get_value()
        match v:
            case Mask(value, flag):

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
