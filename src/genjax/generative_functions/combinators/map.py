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

"""This module implements a generative function combinator which allows
broadcasting for generative functions -- mapping over vectorial versions of
their arguments."""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Trace
from genjax.core.specialization import concrete_cond
from genjax.core.typing import FloatTensor
from genjax.core.typing import IntegerTensor
from genjax.generative_functions.combinators.combinator_datatypes import (
    VectorChoiceMap,
)
from genjax.generative_functions.combinators.combinator_tracetypes import (
    VectorTraceType,
)


#####
# MapTrace
#####


@dataclass
class MapTrace(Trace):
    gen_fn: GenerativeFunction
    indices: IntegerTensor
    inner: Trace
    score: FloatTensor

    def flatten(self):
        return (
            self.indices,
            self.inner,
            self.score,
        ), (self.gen_fn,)

    def get_args(self):
        return self.inner.get_args()

    def get_choices(self):
        return VectorChoiceMap.new(self.indices, self.inner)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.inner.get_retval()

    def get_score(self):
        return self.score


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

    in_args: :code:`Tuple[Int, ...]`
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
                    key, 0.0, 1.0
            )
            key, noise2 = genjax.trace("noise2", genjax.Normal)(
                    key, 0.0, 1.0
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

    # This is a terrible and needs to be re-written.
    # Why do I need to `vmap` to get the correct trace type
    # from the inner kernel? Fix.
    def get_trace_type(self, keys, args, **kwargs):
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]
        keys = jtu.tree_map(lambda v: jnp.zeros(v.shape, v.dtype), keys)
        args = jtu.tree_map(lambda v: jnp.zeros(v.shape, v.dtype), args)
        kernel_tt = jax.vmap(
            self.kernel.get_trace_type, in_axes=(key_axis, arg_axes)
        )(keys, args)
        kernel_tt = jtu.tree_map(lambda v: v[0], kernel_tt)
        return VectorTraceType(kernel_tt, len(keys))

    def simulate(self, key, args, **kwargs):
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]
        indices = np.array([i for i in range(0, len(key))])
        key, tr = jax.vmap(
            self.kernel.simulate,
            in_axes=(key_axis, arg_axes),
        )(key, args)
        map_tr = MapTrace(
            self,
            indices,
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, map_tr

    def _bounds_checker(self, v, key_len):
        lengths = []

        def _inner(v):
            if v.shape[-1] > key_len:
                raise Exception("Length of leaf longer than max length.")
            else:
                lengths.append(v.shape[-1])
                return v

        ret = jtu.tree_map(_inner, v)
        fixed_len = set(lengths)
        assert len(fixed_len) == 1
        return ret, fixed_len.pop()

    # This pads the leaves of a choice map up to
    # `key_len` -- so that we can vmap
    # over the leading axes of the leaves.
    def _padder(self, v, key_len):
        ndim = len(v.shape)
        pad_axes = list(
            (0, key_len - len(v)) if k == 0 else (0, 0) for k in range(0, ndim)
        )
        return (
            np.pad(v, pad_axes)
            if isinstance(v, np.ndarray)
            else jnp.pad(v, pad_axes)
        )

    def _importance_vcm(self, key, chm, args):
        # Get static axes.
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]

        def _importance(key, chm, args):
            return self.kernel.importance(key, chm, args)

        def _simulate(key, chm, args):
            key, tr = self.kernel.simulate(key, args)
            return key, (0.0, tr)

        def _inner(key, index, chm, args):
            check = index == chm.get_index()
            return concrete_cond(
                check,
                _importance,
                _simulate,
                key,
                chm,
                args,
            )

        indices = np.array([i for i in range(0, len(key))])
        key, (w, tr) = jax.vmap(_inner, in_axes=(key_axis, 0, 0, arg_axes))(
            key,
            indices,
            chm,
            args,
        )

        w = jnp.sum(w)
        map_tr = MapTrace(
            self,
            indices,
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr)

    def _importance_fallback(self, key, chm, args):
        # Get static axes.
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]

        # Check incoming choice map, and coerce to `VectorChoiceMap`
        # before passing into scan calls.
        chm, fixed_len = self._bounds_checker(chm, len(key))
        chm = jtu.tree_map(
            lambda chm: self._padder(chm, len(key)),
            chm,
        )
        if not isinstance(chm, VectorChoiceMap):
            indices = np.array(
                [ind if ind < fixed_len else -1 for ind in range(0, len(key))]
            )
            chm = VectorChoiceMap.new(
                indices,
                chm,
            )

        def _importance(key, chm, args):
            return self.kernel.importance(key, chm, args)

        def _simulate(key, chm, args):
            key, tr = self.kernel.simulate(key, args)
            return key, (0.0, tr)

        def _inner(key, index, chm, args):
            check = index == chm.get_index()
            return concrete_cond(
                check,
                _importance,
                _simulate,
                key,
                chm,
                args,
            )

        indices = np.array([i for i in range(0, len(key))])
        key, (w, tr) = jax.vmap(_inner, in_axes=(key_axis, 0, 0, arg_axes))(
            key,
            indices,
            chm,
            args,
        )

        w = jnp.sum(w)
        map_tr = MapTrace(
            self,
            indices,
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, (w, map_tr)

    def importance(self, key, chm, args):
        if isinstance(chm, VectorChoiceMap):
            return self._importance_vcm(key, chm, args)
        else:
            return self._importance_fallback(key, chm, args)

    # The choice map passed in here is a vector choice map.
    def _update_vcm(self, key, prev, chm, diffs):
        def _update(key, prev, chm, diffs):
            key, (retdiff, w, tr, d) = self.kernel.update(
                key, prev, chm, diffs
            )
            return key, (retdiff, w, tr, d)

        def _fallback(key, prev, chm, diffs):
            key, (retdiff, w, tr, d) = self.kernel.update(
                key, prev, EmptyChoiceMap(), diffs
            )
            return key, (retdiff, w, tr, d)

        def _inner(key, index, prev, chm, diffs):
            check = index == chm.get_index()
            return concrete_cond(
                check,
                _update,
                _fallback,
                key,
                prev,
                chm,
                diffs,
            )

        # Get static axes.
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]

        indices = np.array([i for i in range(0, len(key))])
        prev_inaxes_tree = jtu.tree_map(
            lambda v: None if v.shape == () else 0,
            prev.inner,
        )
        key, (retdiff, w, tr, discard) = jax.vmap(
            _inner,
            in_axes=(key_axis, 0, prev_inaxes_tree, 0, arg_axes),
        )(key, indices, prev.inner, chm, diffs)

        w = jnp.sum(w)
        map_tr = MapTrace(
            self,
            indices,
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, (retdiff, w, map_tr, discard)

    # The choice map doesn't carry optimization info.
    def _update_fallback(self, key, prev, chm, diffs):
        def _update(key, prev, chm, diffs):
            key, (retdiff, w, tr, d) = self.kernel.update(
                key, prev, chm, diffs
            )
            return key, (retdiff, w, tr, d)

        def _fallback(key, prev, chm, diffs):
            key, (retdiff, w, tr, d) = self.kernel.update(
                key, prev, EmptyChoiceMap(), diffs
            )
            return key, (retdiff, w, tr, d)

        def _inner(key, index, prev, chm, diffs):
            check = index == chm.get_index()
            return concrete_cond(
                check,
                _update,
                _fallback,
                key,
                prev,
                chm,
                diffs,
            )

        # Get static axes.
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]

        # Check incoming choice map, and coerce to `VectorChoiceMap`
        # before passing into scan calls.
        chm, fixed_len = self._bounds_checker(chm, len(key))
        chm = jtu.tree_map(
            lambda chm: self._padder(chm, len(key)),
            chm,
        )
        if not isinstance(chm, VectorChoiceMap):
            indices = np.array(
                [ind if ind < fixed_len else -1 for ind in range(0, len(key))]
            )
            chm = VectorChoiceMap.new(
                indices,
                chm,
            )

        # Now, we proceed.
        indices = np.array([i for i in range(0, len(key))])
        prev_inaxes_tree = jtu.tree_map(
            lambda v: None if v.shape == () else 0,
            prev.inner,
        )
        key, (retdiff, w, tr, discard) = jax.vmap(
            _inner,
            in_axes=(key_axis, 0, prev_inaxes_tree, 0, arg_axes),
        )(key, indices, prev.inner, chm, diffs)

        w = jnp.sum(w)
        map_tr = MapTrace(
            self,
            indices,
            tr,
            jnp.sum(tr.get_score()),
        )

        return key, (retdiff, w, map_tr, discard)

    def update(self, key, prev, chm, diffs):
        assert isinstance(prev, MapTrace)

        # Branches here implement certain optimizations when more
        # information about the passed in choice map is available.
        #
        # The fallback just inflates a choice map to the right shape
        # and runs a generic update.
        if isinstance(chm, VectorChoiceMap):
            return self._update_vcm(key, prev, chm, diffs)
        else:
            return self._update_fallback(key, prev, chm, diffs)

    def _throw_index_check_host_exception(self, index: int):
        def _inner(count, transforms):
            raise Exception(
                f"\nMapCombinator {self} received a choice map with mismatched indices (at index {index}) in assess."
            )

        hcb.id_tap(
            lambda *args: _inner(*args),
            index,
            result=None,
        )
        return None

    def assess(self, key, chm, args):
        assert isinstance(chm, VectorChoiceMap)

        def _inner(key, index, chm, args):
            check = index == chm.get_index()

            # This inserts a host callback check for bounds checking.
            # If there is an index failure, `assess` must fail
            # because we must provide a constraint for every generative
            # function call.
            concrete_cond(
                check,
                lambda *args: self._throw_index_check_host_exception(
                    index,
                ),
                lambda *args: None,
            )

            return self.kernel.assess(key, chm, args)

        # Get static axes.
        key_axis = self.in_axes[0]
        arg_axes = self.in_axes[1:]

        indices = np.array([i for i in range(0, len(key))])
        return jax.vmap(_inner, in_axes=(key_axis, 0, 0, arg_axes))(
            key,
            indices,
            chm,
            args,
        )
