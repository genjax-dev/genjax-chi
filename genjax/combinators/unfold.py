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
This module implements a generative function combinator which allows
statically unrolled control flow for generative functions which can act
as kernels (accepting their previous output as input).
"""

import jax
import jax.numpy as jnp
from genjax.core.datatypes import GenerativeFunction, Trace
from dataclasses import dataclass
from typing import Any, Tuple

#####
# UnfoldTrace
#####


@dataclass
class UnfoldTrace(Trace):
    gen_fn: GenerativeFunction
    mapped_subtrace: Trace
    args: Tuple
    retval: Any
    score: jnp.float32

    def get_args(self):
        return self.args

    def get_choices(self):
        return self.mapped_subtrace

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def flatten(self):
        return (
            self.mapped_subtrace,
            self.args,
            self.retval,
            self.score,
        ), (self.gen_fn,)

    @classmethod
    def unflatten(cls, data, xs):
        return UnfoldTrace(*data, *xs)


#####
# UnfoldCombinator
#####


@dataclass
class UnfoldCombinator(GenerativeFunction):
    kernel: GenerativeFunction
    length: int

    def flatten(self):
        return (self.length,), (self.kernel,)

    @classmethod
    def unflatten(cls, data, xs):
        return UnfoldCombinator(*data, *xs)

    def __call__(self, key, *args):
        def __inner(key, *args, x):
            return self.kernel(key, *args)

        return jax.lax.scan(
            __inner,
            (key, args),
            None,
            length=self.length,
        )

    def simulate(self, key, args):
        def __inner(carry, x):
            key, tr = self.kernel.simulate(*carry)
            retval = tr.get_retval()
            return (key, retval), tr

        (key, retval), tr = jax.lax.scan(
            __inner,
            (key, args),
            None,
            length=self.length,
        )

        unfold_tr = UnfoldTrace(
            self,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        return key, unfold_tr
