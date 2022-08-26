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
from genjax.core.datatypes import ChoiceMap, GenerativeFunction
from genjax.core import Pytree
from dataclasses import dataclass
from typing import Tuple, Callable, Any

#####
# ValueChoiceMap
#####


@dataclass
class ValueChoiceMap(ChoiceMap):
    value: Any

    def __getitem__(self, k):
        assert isinstance(k, tuple)
        assert len(k) == 0
        return self

    def get_value(self):
        return self.value

    def flatten(self):
        return (self.value,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return ValueChoiceMap(*data, *xs)


#####
# DistributionTrace
#####


@dataclass
class DistributionTrace(Pytree):
    gen_fn: Callable
    args: Tuple
    value: Any
    score: Any

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

    def __getitem__(self, k):
        assert isinstance(k, tuple)
        assert len(k) == 0
        return self

    def flatten(self):
        return (self.args, self.value, self.score), (self.gen_fn,)

    @classmethod
    def unflatten(cls, data, xs):
        return DistributionTrace(*data, *xs)


#####
# Distribution
#####


@dataclass
class Distribution(GenerativeFunction):

    # Default implementations of the `Pytree` interfaces.
    def flatten(self):
        return (), ()

    @classmethod
    def unflatten(cls, values, slices):
        return cls()

    @classmethod
    @abc.abstractmethod
    def abstract_eval(cls, key, p, shape=()):
        pass

    @classmethod
    @abc.abstractmethod
    def sample(cls, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def logpdf(cls, v, *args, **kwargs):
        pass

    def simulate(self, key, args, **kwargs):
        key, sub_key = jax.random.split(key)
        v = self.sample(sub_key, *args, **kwargs)
        score = self.logpdf(v, *args)
        tr = DistributionTrace(self, args, v, score)
        return (key, tr)

    def importance(self, key, chm, args, **kwargs):
        chm = chm.get_choices()
        v = chm.get_value()
        weight = self.logpdf(v, *args)
        return key, (weight, DistributionTrace(self, args, v, weight))

    def diff(self, key, prev, new, args, **kwargs):
        bwd = prev.get_score()
        v = new.get_value()
        fwd = self.logpdf(v, *args, **kwargs)
        return key, (fwd - bwd, (v,))

    def update(self, key, original, chm, args, **kwargs):
        chm = chm.get_choices()
        old_weight = original.get_score()
        assert isinstance(chm, ValueChoiceMap)
        v = chm.get_value()
        weight = self.logpdf(v, *args)
        return key, (
            weight - old_weight,
            DistributionTrace(self, args, v, weight),
            original.get_choices(),
        )
