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
broadcasting for generative functions -- mapping over vectorial versions of their arguments.
"""

import jax
import jax.numpy as jnp
from genjax.core.datatypes import EmptyChoiceMap, GenerativeFunction, Trace
from genjax.interface import (
    sample,
    simulate,
    importance,
    diff,
    update,
    arg_grad,
    choice_grad,
)
from dataclasses import dataclass
from typing import Tuple

#####
# MapTrace
#####


@dataclass
class MapTrace(Trace):
    gen_fn: GenerativeFunction
    mapped_subtrace: Trace
    args: Tuple
    score: jnp.float32

    def get_args(self):
        return self.args

    def get_choices(self):
        return self.mapped_subtrace

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.mapped_subtrace.get_retval()

    def get_score(self):
        return self.score

    def flatten(self):
        return (
            self.mapped_subtrace,
            self.args,
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
            tr,
            args,
            jnp.sum(tr.get_score()),
        )

        return key, map_tr

    def _importance(self, key, chm, args, index, **kwargs):
        if chm.has_key(index):
            chm = chm.get_key(index)
        else:
            chm = EmptyChoiceMap()
        key, (w, tr) = importance(self.kernel)(key, chm, args)
        return key, (w, tr)

    def importance(self, key, chm, args, **kwargs):
        key, (w, tr) = jax.vmap(self._importance, in_axes=(0, None, 0, 0))(
            key, chm, args
        )
        map_tr = MapTrace(
            self,
            tr,
            args,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr)

    def diff(self, key, original, new, args, **kwargs):
        in_axes = kwargs["in_axes"]
        key, (w, tr) = jax.vmap(diff(self.kernel), in_axes=in_axes)(
            key, original, new, args
        )
        map_tr = MapTrace(
            self,
            tr,
            args,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr)

    def update(self, key, original, new, args, **kwargs):
        in_axes = kwargs["in_axes"]
        key, (w, tr, discard) = jax.vmap(update(self.kernel), in_axes=in_axes)(
            key, original, new, args
        )
        map_tr = MapTrace(
            self,
            tr,
            args,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr, discard)

    def arg_grad(self, key, tr, args, **kwargs):
        in_axes = kwargs["in_axes"]
        key, arg_grads = jax.vmap(arg_grad(self.kernel), in_axes=in_axes)(
            key, tr, args
        )
        return key, arg_grads

    def choice_grad(self, key, tr, chm, args, **kwargs):
        in_axes = kwargs["in_axes"]
        key, choice_grads = jax.vmap(choice_grad(self.kernel), in_axes=in_axes)(
            key, tr, chm, args
        )
        return key, choice_grads
