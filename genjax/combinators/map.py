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
from genjax.builtin.shape_analysis import choice_map_shape
from genjax.core.datatypes import (
    GenerativeFunction,
    Trace,
    BooleanMask,
)
from dataclasses import dataclass
from .vector_choice_map import VectorChoiceMap, prepare_vectorized_choice_map
from typing import Tuple

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
        return VectorChoiceMap(self.subtrace)

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
    """
    :code:`MapCombinator` accepts a single generative function as input and
    provides :code:`vmap`-based implementations of the generative function
    interface methods. :code:`MapCombinator` also accepts :code:`in_axes` as
    an argument to specify exactly which axes of the :code:`(key, *args)`
    should be broadcasted over.

    Parameters
    ----------

    gen_fn: :code:`GenerativeFunction`
        A single `GenerativeFunction` instance.

    in_args: :code:`Tuple[Int]`
        A tuple specifying which :code:`(key, *args)` to broadcast
        over.

    Returns
    -------
    :code:`MapCombinator`
        A single :code:`MapCombinator` generative function which
        implements :code:`vmap` support for each generative function
        interface method.

    Example
    -------

    .. jupyter-execute::

        import jax
        import jax.numpy as jnp
        import genjax

        @genjax.gen
        def add_normal_noise(key, x):
            key, noise1 = genjax.trace("noise1", genjax.Normal)(
                    key, (0.0, 1.0)
            )
            key, noise2 = genjax.trace("noise2", genjax.Normal)(
                    key, (0.0, 1.0)
            )
            return (key, x + noise1 + noise2)


        mapped = genjax.MapCombinator(add_normal_noise, in_axes=(0, 0))

        arr = jnp.ones(100)
        key = jax.random.PRNGKey(314159)
        key, *subkeys = jax.random.split(key, 101)
        subkeys = jnp.array(subkeys)
        _, tr = jax.jit(genjax.simulate(mapped))(subkeys, (arr, ))
        print(tr)
    """

    kernel: GenerativeFunction
    in_axes: Tuple

    def flatten(self):
        return (), (self.kernel, self.in_axes)

    @classmethod
    def unflatten(cls, data, xs):
        return MapCombinator(*data, *xs)

    def __call__(self, key, *args, **kwargs):
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]
        vmapped = jax.vmap(
            self.kernel.simulate,
            in_axes=(key_axis, arg_axes),
        )
        key, tr = vmapped(key, args)
        retval = tr.get_retval()
        return key, retval

    def simulate(self, key, args, **kwargs):
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]
        key, tr = jax.vmap(
            self.kernel.simulate,
            in_axes=(key_axis, arg_axes),
        )(key, args)
        map_tr = MapTrace(
            self,
            len(key),
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, map_tr

    def importance(self, key, chm, args):
        if not isinstance(chm, VectorChoiceMap) and not isinstance(
            chm, BooleanMask
        ):
            length = len(key)  # static
            assert length > 0
            _, treedef, shape = choice_map_shape(self.kernel)(key[0], args)
            chm_vectored, mask_vectored = prepare_vectorized_choice_map(
                shape, treedef, length, chm
            )

            chm = BooleanMask(chm_vectored, mask_vectored)
        if isinstance(chm, VectorChoiceMap):
            chm = chm.subtrace

        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]
        key, (w, tr) = jax.vmap(
            self.kernel.importance, in_axes=(key_axis, 0, arg_axes)
        )(
            key,
            chm,
            args,
        )

        w = jnp.sum(w)
        map_tr = MapTrace(
            self,
            len(key),
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr)

    def update(self, key, prev, chm, args):
        if not isinstance(chm, VectorChoiceMap) or not isinstance(
            chm, BooleanMask
        ):
            length = len(key)  # static
            assert length > 0
            _, treedef, shape = choice_map_shape(self.kernel)(key[0], args)
            chm_vectored, mask_vectored = prepare_vectorized_choice_map(
                shape, treedef, length, chm
            )

            chm = BooleanMask(chm_vectored, mask_vectored)

        # The previous trace has to have a VectorChoiceMap
        # here.
        prev = prev.get_choices()
        assert isinstance(prev, VectorChoiceMap)
        prev = prev.subtrace

        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]
        key, (w, tr, discard) = jax.vmap(
            self.kernel.update, in_axes=(key_axis, 0, 0, arg_axes)
        )(key, prev, chm, args)

        w = jnp.sum(w)
        map_tr = MapTrace(
            self,
            len(key),
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr, discard)
