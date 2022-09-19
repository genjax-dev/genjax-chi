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

"""
This module contains the `Distribution` abstact base class.
"""

import abc
import jax
from genjax.core.datatypes import (
    EmptyChoiceMap,
    ValueChoiceMap,
    Trace,
    GenerativeFunction,
    AllSelection,
)
from genjax.core.masks import BooleanMask
from genjax.core.specialization import (
    concrete_cond,
    concrete_and,
)
from genjax.core.tracetypes import Bottom
from dataclasses import dataclass
from typing import Tuple, Callable, Any

#####
# DistributionTrace
#####


@dataclass
class DistributionTrace(Trace):
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

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return (self.value.get_leaf_value(),)

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

    def __call__(self, key, *args, **kwargs):
        key, subkey = jax.random.split(key)
        v = self.sample(subkey, *args, **kwargs)
        return (key, v)

    def __trace_type__(self, key, *args, **kwargs):
        shape = kwargs.get("shape", ())
        return Bottom(shape)

    def get_trace_type(self, key, args, **kwargs):
        return self.__trace_type__(key, *args, **kwargs)

    @classmethod
    @abc.abstractmethod
    def sample(cls, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def logpdf(cls, key, v, *args, **kwargs):
        pass

    def simulate(self, key, args, **kwargs):
        key, sub_key = jax.random.split(key)
        v = self.sample(sub_key, *args, **kwargs)
        key, sub_key = jax.random.split(key)
        score = self.logpdf(sub_key, v, *args)
        tr = DistributionTrace(self, args, ValueChoiceMap(v), score)
        return key, tr

    @BooleanMask.collapse_boundary
    def importance(self, key, chm, args, **kwargs):
        def _importance_branch(key, chm, args):
            v = chm.get_leaf_value()
            key, sub_key = jax.random.split(key)
            w = self.logpdf(sub_key, v, *args)
            return key, v, w, w

        def _simulate_branch(key, chm, args):
            key, sub_key = jax.random.split(key)
            v = self.sample(sub_key, *args, **kwargs)
            w = 0.0
            key, sub_key = jax.random.split(key)
            score = self.logpdf(sub_key, v, *args)
            return key, v, w, score

        key, v, w, score = concrete_cond(
            chm.is_leaf(),
            _importance_branch,
            _simulate_branch,
            key,
            chm,
            args,
        )
        return key, (
            w,
            DistributionTrace(self, args, ValueChoiceMap(v), score),
        )

    @BooleanMask.collapse_boundary
    def update(self, key, prev, new, args, **kwargs):
        assert isinstance(prev, DistributionTrace)

        has_previous = prev.is_leaf()
        constrained = new.is_leaf()

        def _update_branch(key, args):
            prev_score = prev.get_score()
            v = new.get_leaf_value()
            key, sub_key = jax.random.split(key)
            fwd = self.logpdf(sub_key, v, *args)
            discard = BooleanMask.new(True, prev.get_choices())
            return key, (fwd - prev_score, v, discard)

        def _has_prev_branch(key, args):
            v = prev.get_leaf_value()
            discard = BooleanMask.new(False, prev.get_choices())
            return key, (0.0, v, discard)

        def _constrained_branch(key, args):
            chm = prev.get_choices()
            key, (w, tr) = self.importance(key, chm, args)
            v = tr.get_leaf_value()
            discard = BooleanMask.new(False, prev.get_choices())
            return key, (w, v, discard)

        key, (w, v, discard) = concrete_cond(
            concrete_and(has_previous, constrained),
            _update_branch,
            lambda key, args: concrete_cond(
                has_previous, _has_prev_branch, _constrained_branch, key, args
            ),
            key,
            args,
        )

        # This ensures Pytree type consistency.
        # The weight, values, etc -- are all computed
        # correctly (dynamically), but we have to ensure
        # that leaves which are returned have the same type
        # as leaves which come in.
        if isinstance(prev.get_choices(), BooleanMask):
            mask = prev.get_choices().mask
            vchm = BooleanMask(mask, ValueChoiceMap(v))
        else:
            vchm = ValueChoiceMap(v)

        return key, (
            w,
            DistributionTrace(self, args, vchm, w),
            discard,
        )
