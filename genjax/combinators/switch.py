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

import jax
import jax.numpy as jnp
from genjax.core.datatypes import ChoiceMap, Trace, GenerativeFunction
from dataclasses import dataclass
from typing import Tuple, Any

#####
# SwitchChoiceMap
#####


@dataclass
class SwitchChoiceMap(ChoiceMap):
    branch: bool
    branch_1: ChoiceMap
    branch_2: ChoiceMap

    def get_choices(self):
        if self.branch:
            return self.branch_1
        else:
            return self.branch_2

    def flatten(self):
        return (self.branch, self.branch_1, self.branch_2), ()

    @classmethod
    def unflatten(cls, xs, data):
        return SwitchChoiceMap(*xs, *data)


#####
# SwitchTrace
#####


@dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: SwitchChoiceMap
    score: jnp.float32

    def get_choices(self):
        return self.choices

    def get_args(self):
        return self.args

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def flatten(self):
        return (self.args, self.retval, self.choices, self.score), (
            self.gen_fn,
        )

    @classmethod
    def unflatten(cls, data, xs):
        return SwitchTrace(*data, *xs)


#####
# SwitchCombinator
#####


def _fill(shape, dtype):
    if dtype == bool:
        return [True for _ in shape]
    else:
        return [0.0 for _ in shape]


@dataclass
class SwitchCombinator(GenerativeFunction):
    branch_1: GenerativeFunction
    branch_2: GenerativeFunction

    def __init__(self, branch1, branch2):
        self.branch_1 = branch1
        self.branch_2 = branch2
        self.blank_1 = None
        self.blank_2 = None

    def flatten(self):
        return (), (self.branch_1, self.branch_2)

    @classmethod
    def unflatten(cls, xs, data):
        return SwitchCombinator(*xs, *data)

    def __call__(self, key, *args):
        return jax.lax.cond(
            args[0],
            lambda *args: self.branch_1(key, *args),
            lambda *args: self.branch_2(key, *args),
            *args[1:],
        )

    def _simulate_with_switch_map_1(self, gen_fn, blank, key, args):
        key, tr = gen_fn.simulate(key, args)
        chm = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        sw_chm = SwitchChoiceMap(False, chm, blank)
        return key, SwitchTrace(
            self,
            args,
            retval,
            sw_chm,
            score,
        )

    def _simulate_with_switch_map_2(self, gen_fn, blank, key, args):
        key, tr = gen_fn.simulate(key, args)
        chm = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        sw_chm = SwitchChoiceMap(True, blank, chm)
        return key, SwitchTrace(
            self,
            args,
            retval,
            sw_chm,
            score,
        )

    def simulate(self, key, args):
        switch = args[0]

        # Create a blank filled Pytree for branch_1.
        jaxpr_1, ret = jax.make_jaxpr(
            self.branch_1.simulate, return_shape=True
        )(key, args[1:])
        chm_shape = ret[1].get_choices()
        leaves, form = jax.tree_util.tree_flatten(chm_shape)
        blank_1 = jax.tree_map(
            lambda k: jax.numpy.zeros(k.shape, dtype=k.dtype), chm_shape
        )

        # Create a blank filled Pytree for branch_2.
        jaxpr_2, ret = jax.make_jaxpr(
            self.branch_2.simulate, return_shape=True
        )(key, args[1:])
        chm_shape = ret[1].get_choices()
        leaves, form = jax.tree_util.tree_flatten(chm_shape)
        blank_2 = jax.tree_map(
            lambda k: jax.numpy.zeros(k.shape, dtype=k.dtype), chm_shape
        )

        return jax.lax.cond(
            switch,
            lambda *args: self._simulate_with_switch_map_1(
                self.branch_1,
                blank_2,
                key,
                args,
            ),
            lambda *args: self._simulate_with_switch_map_2(
                self.branch_2,
                blank_1,
                key,
                args,
            ),
            *args[1:],
        )
