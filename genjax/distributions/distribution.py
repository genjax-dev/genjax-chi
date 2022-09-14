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
from genjax.core.specialization import concrete_cond
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
    value: Any
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
        return (self.value,)

    def get_args(self):
        return self.args

    def get_score(self):
        return self.score

    def get_choices(self):
        return ValueChoiceMap(self.value)

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
        tr = DistributionTrace(self, args, v, score)
        return key, tr

    def importance(self, key, chm, args, **kwargs):
        chm = chm.get_choices()

        def _simulate_branch(key, chm, args):
            key, sub_key = jax.random.split(key)
            v = self.sample(sub_key, *args, **kwargs)
            w = 0.0
            key, sub_key = jax.random.split(key)
            score = self.logpdf(sub_key, v, *args)
            return key, v, w, score

        def _importance_branch(key, chm, args):
            v = chm.get_value()
            key, sub_key = jax.random.split(key)
            w = self.logpdf(sub_key, v, *args)
            return key, v, w, w

        key, v, w, score = concrete_cond(
            chm.has_value(),
            _importance_branch,
            _simulate_branch,
            key,
            chm,
            args,
        )

        return key, (w, DistributionTrace(self, args, v, score))

    def update(self, key, prev, new, args, **kwargs):
        has_previous = prev.has_value()
        constrained = new.has_value()

        def _update_branch(key, args):
            prev_score = prev.get_score()
            v = new.get_value()
            key, sub_key = jax.random.split(key)
            fwd = self.logpdf(sub_key, v, *args)
            discard = BooleanMask(prev.get_choices(), True)
            return key, (fwd - prev_score, v, discard)

        def _has_prev_branch(key, args):
            v = prev.get_value()
            discard = BooleanMask(prev.get_choices(), False)
            return key, (0.0, v, discard)

        def _constrained_branch(key, args):
            chm = new.get_choice(())
            key, (w, tr) = self.importance(key, chm, args)
            v = tr.get_value()
            discard = BooleanMask(prev.get_choices(), False)
            return key, (w, v, discard)

        key, (w, v, discard) = concrete_cond(
            has_previous * constrained,
            _update_branch,
            lambda key, args: concrete_cond(
                has_previous, _has_prev_branch, _constrained_branch, key, args
            ),
            key,
            args,
        )

        return key, (w, DistributionTrace(self, args, v, w), discard)
