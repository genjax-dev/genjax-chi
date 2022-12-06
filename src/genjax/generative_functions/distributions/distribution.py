# Copyright 2022 MIT Probabilistic Computing Project
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
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Tuple

import jax
import jax.numpy as jnp

from genjax.core.datatypes import AllSelection
from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Trace
from genjax.core.datatypes import ValueChoiceMap
from genjax.core.masks import BooleanMask
from genjax.core.specialization import concrete_cond
from genjax.core.tree import Leaf
from genjax.generative_functions.builtin.builtin_tracetype import lift
from genjax.generative_functions.diff_rules import Diff
from genjax.generative_functions.diff_rules import NoChange
from genjax.generative_functions.diff_rules import check_no_change
from genjax.generative_functions.diff_rules import strip_diff


#####
# DistributionTrace
#####


@dataclass
class DistributionTrace(Trace, Leaf):
    gen_fn: Callable
    args: Tuple
    value: ValueChoiceMap
    score: Any

    def flatten(self):
        return (self.args, self.value, self.score), (self.gen_fn,)

    def project(self, selection):
        if isinstance(selection, AllSelection):
            return self.get_choices(), self.score
        else:
            return EmptyChoiceMap(), 0.0

    def get_leaf_value(self):
        return self.value.get_leaf_value()

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.value.get_leaf_value()

    def get_args(self):
        return self.args

    def get_score(self):
        return self.score

    def get_choices(self):
        return self.value

    def merge(self, other):
        return other


#####
# Distribution
#####


@dataclass
class Distribution(GenerativeFunction):
    def flatten(self):
        return (), ()

    def get_trace_type(self, key, args, **kwargs):
        _, (_, (_, ttype)) = jax.make_jaxpr(
            self.random_weighted, return_shape=True
        )(key, *args)
        return lift(ttype)

    @abc.abstractmethod
    def random_weighted(self, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def estimate_logpdf(cls, key, v, *args, **kwargs):
        pass

    def simulate(self, key, args, **kwargs):
        key, (w, v) = self.random_weighted(key, *args, **kwargs)
        tr = DistributionTrace(self, args, ValueChoiceMap(v), w)
        return key, tr

    def importance(self, key, chm, args, **kwargs):
        assert isinstance(chm, ChoiceMap)
        assert isinstance(chm, Leaf)

        # If the choice map is empty, we just simulate
        # and return 0.0 for the log weight.
        if isinstance(chm, EmptyChoiceMap):
            key, tr = self.simulate(key, args, **kwargs)
            return key, (0.0, tr)

        # If it's not empty, we should check if it is a mask.
        # If it is a mask, we need to see if it is active or not,
        # and then unwrap it - and use the active flag to determine
        # what to do at runtime.
        v = chm.get_leaf_value()
        if isinstance(v, BooleanMask):
            active = v.mask()
            v = v.unmask()

            def _active(key, v, args):
                key, (w, v) = self.estimate_logpdf(key, v, *args)
                return key, v, w

            def _inactive(key, v, args):
                w = 0.0
                return key, v, w

            key, v, w = concrete_cond(
                active,
                _active,
                _inactive,
                key,
                v,
                args,
            )
            score = w

        # Otherwise, we just estimate the logpdf of the value
        # we got out of the choice map.
        else:
            key, (w, v) = self.estimate_logpdf(key, v, *args)
            score = w

        return key, (
            w,
            DistributionTrace(self, args, ValueChoiceMap(v), score),
        )

    def update(self, key, prev, new, diffs, **kwargs):
        assert isinstance(prev, DistributionTrace)
        assert isinstance(new, ChoiceMap)

        # Incremental optimization - if nothing has changed,
        # just return the previous trace.
        if isinstance(new, EmptyChoiceMap) and all(
            map(check_no_change, diffs)
        ):
            v = prev.get_retval()
            retval_diff = Diff.new(v, change=NoChange)
            return key, (retval_diff, 0.0, prev, EmptyChoiceMap())

        # Otherwise, we consider the cases.
        args = tuple(map(strip_diff, diffs))

        # First, we have to check if the trace provided
        # is masked or not. It's possible that a trace
        # with a mask is updated.
        prev_v = prev.get_retval()
        active = True
        if isinstance(prev_v, BooleanMask):
            prev_v = prev_v.unmask()
            active = prev_v.mask

        # Case 1: the new choice map is empty here.
        if isinstance(new, EmptyChoiceMap):
            prev_score = prev.get_score()
            v = prev_v

            # If the value is active, we compute any weight
            # corrections from changing arguments.
            def _active(key, v, *args):
                key, (fwd, _) = self.estimate_logpdf(key, v, *args)
                return key, fwd - prev_score

            # If the value is inactive, we do nothing.
            def _inactive(key, v, *args):
                return key, prev_score

            key, w = concrete_cond(active, _active, _inactive, key, v, *args)
            discard = EmptyChoiceMap()
            retval_diff = Diff.new(prev.get_retval(), change=NoChange)

        # Case 2: the new choice map is not empty here.
        else:
            prev_score = prev.get_score()
            v = new.get_leaf_value()

            # Now, we must check if the choice map has a masked
            # leaf value, and dispatch accordingly.
            active_chm = True
            if isinstance(v, BooleanMask):
                v = v.unmask()
                active_chm = v.mask

            # The only time this flag is on is when both leaf values
            # are concrete, or they are both masked with true mask
            # values.
            active = jnp.all(jnp.logical_and(active_chm, active))

            def _new_active(key, v, *args):
                key, (fwd, _) = self.estimate_logpdf(key, v, *args)
                return key, v, fwd - prev_score

            def _new_inactive(key, v, *args):
                return key, prev_v, 0.0

            key, v, w = concrete_cond(
                active_chm, _new_active, _new_inactive, key, v, *args
            )

            # Now, we also must check if the trace has a masked
            # leaf value. If it does, we decide what to do.

            discard = ValueChoiceMap(prev.get_leaf_value())
            retval_diff = Diff.new(v)

        return key, (
            retval_diff,
            w,
            DistributionTrace(self, args, ValueChoiceMap(v), w),
            discard,
        )

    def assess(self, key, evaluation_point, args):
        assert isinstance(evaluation_point, ValueChoiceMap)
        v = evaluation_point.get_leaf_value()
        key, (score, _) = self.estimate_logpdf(key, v, *args)
        return key, (v, score)


#####
# ExactDensity
#####


@dataclass
class ExactDensity(Distribution):
    @abc.abstractmethod
    def sample(self, key, *args, **kwargs):
        pass

    @abc.abstractmethod
    def logpdf(self, v, *args, **kwargs):
        pass

    def random_weighted(self, key, *args, **kwargs):
        key, sub_key = jax.random.split(key)
        v = self.sample(sub_key, *args, **kwargs)
        w = self.logpdf(v, *args, **kwargs)
        return key, (w, v)

    def estimate_logpdf(self, key, v, *args, **kwargs):
        w = self.logpdf(v, *args, **kwargs)
        return key, (w, v)
