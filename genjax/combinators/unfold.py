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
import jax.tree_util as jtu
from genjax.core.datatypes import (
    GenerativeFunction,
    Trace,
)
from genjax.builtin.shape_analysis import choice_map_shape
from genjax.interface import simulate, importance
from dataclasses import dataclass
from typing import Any, Tuple
from .dynamic import DynamicJAXGenerativeFunction
from .vector_choice_map import VectorChoiceMap, prepare_vectorized_choice_map

#####
# UnfoldTrace
#####


@dataclass
class UnfoldTrace(Trace):
    gen_fn: GenerativeFunction
    length: int
    subtrace: Trace
    args: Tuple
    retval: Any
    score: jnp.float32

    def get_args(self):
        return self.args

    def get_choices(self):
        _, form = jtu.tree_flatten(self.subtrace)
        return VectorChoiceMap(self.subtrace, self.length, form)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def flatten(self):
        return (
            self.length,
            self.subtrace,
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
            key, tr = simulate(self.kernel)(*carry)
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
            self.length,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        return key, unfold_tr

    def importance(self, key, chm, args):
        length = self.length
        assert length > 0
        _, treedef, shape = choice_map_shape(self.kernel)(key, args)
        chm_vectored, mask_vectored = prepare_vectorized_choice_map(
            shape, treedef, length, chm
        )

        dynamic_kernel = DynamicJAXGenerativeFunction(self.kernel)

        # To be or not to be `scan`d.
        def __inner(carry, x):
            key, args = carry
            mask, chm = x
            key, (w, tr) = importance(dynamic_kernel)(key, mask, chm, args)
            retval = tr.get_retval()
            return (key, retval), (w, tr)

        (key, retval), (w, tr) = jax.lax.scan(
            __inner,
            (key, args),
            (mask_vectored, chm_vectored),
            length=self.length,
        )

        unfold_tr = UnfoldTrace(
            self,
            self.length,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr)
