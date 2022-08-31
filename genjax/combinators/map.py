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
broadcasting for generative functions -- mapping over
vectorial versions of their arguments.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from genjax.builtin.shape_analysis import choice_map_shape
from genjax.core.datatypes import (
    GenerativeFunction,
    Trace,
)
from genjax.interface import (
    sample,
    simulate,
)
from dataclasses import dataclass
from .dynamic import DynamicJAXGenerativeFunction
from .vector_choice_map import VectorChoiceMap, prepare_vectorized_choice_map

#####
# MapTrace
#####


@dataclass
class MapTrace(Trace):
    gen_fn: GenerativeFunction
    length: int
    subtrace: Trace
    score: jnp.float32

    def get_args(self):
        return self.subtrace.get_args()

    def get_choices(self):
        _, form = jtu.tree_flatten(self.subtrace)
        return VectorChoiceMap(self.subtrace, self.length, form)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.subtrace.get_retval()

    def get_score(self):
        return self.score

    def flatten(self):
        return (
            self.length,
            self.subtrace,
            self.score,
        ), (self.gen_fn,)

    @classmethod
    def unflatten(cls, data, xs):
        return MapTrace(*data, *xs)


#####
# MapCombinator
#####


@dataclass
class MapCombinator(GenerativeFunction):
    kernel: GenerativeFunction

    def flatten(self):
        return (), (self.kernel,)

    @classmethod
    def unflatten(cls, data, xs):
        return MapCombinator(*data, *xs)

    def __call__(self, key, *args, **kwargs):
        vmapped = jax.vmap(sample(self.kernel), in_axes=(0, 0))
        return vmapped(key, args)

    def simulate(self, key, args, **kwargs):
        key, tr = jax.vmap(simulate(self.kernel), in_axes=(0, 0))(key, args)
        map_tr = MapTrace(
            self,
            len(key),
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, map_tr

    def importance(self, key, chm, args):
        length = len(key)  # static
        assert length > 0
        _, treedef, shape = choice_map_shape(self.kernel)(key[0], args)
        chm_vectored, mask_vectored = prepare_vectorized_choice_map(
            shape, treedef, length, chm
        )

        dynamic_kernel = DynamicJAXGenerativeFunction(self.kernel)

        # To be or not to be `vmap`d.
        def __inner(key, mask, chm, args):
            return dynamic_kernel.importance(key, mask, chm, args)

        key, (w, tr) = jax.vmap(__inner, in_axes=(0, 0, 0, 0))(
            key, mask_vectored, chm_vectored, args
        )

        map_tr = MapTrace(
            self,
            len(key),
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr)
