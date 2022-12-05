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
statically unrolled control flow for generative functions which can act as
kernels (a kernel generative function can accept their previous output as
input)."""

from dataclasses import dataclass
from typing import Any
from typing import Sequence
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
from genjax.generative_functions.combinators.combinator_datatypes import (
    VectorChoiceMap,
)
from genjax.generative_functions.combinators.combinator_tracetypes import (
    VectorTraceType,
)


#####
# UnfoldTrace
#####


@dataclass
class UnfoldTrace(Trace):
    gen_fn: GenerativeFunction
    indices: Sequence
    inner: Trace
    args: Tuple
    retval: Any
    score: jnp.float32

    def flatten(self):
        return (
            self.indices,
            self.inner,
            self.args,
            self.retval,
            self.score,
        ), (self.gen_fn,)

    def get_args(self):
        return self.args

    def get_choices(self):
        return VectorChoiceMap(self.indices, self.inner)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score


#####
# UnfoldCombinator
#####


@dataclass
class UnfoldCombinator(GenerativeFunction):
    """
    :code:`UnfoldCombinator` accepts a single kernel generative function
    as input and a static unroll length which specifies how many iterations
    to run the chain for.

    A kernel generative function is one which accepts and returns
    the same signature of arguments. Under the hood, :code:`UnfoldCombinator`
    is implemented using :code:`jax.lax.scan` - which has the same
    requirements.

    Parameters
    ----------

    gen_fn: :code:`GenerativeFunction`
        A single *kernel* `GenerativeFunction` instance.

    length: :code:`Int`
        An integer specifying the unroll length of the chain of applications.

    Returns
    -------
    :code:`UnfoldCombinator`
        A single :code:`UnfoldCombinator` generative function which
        implements the generative function interface using a scan-like
        pattern. This generative function will perform a dependent-for
        iteration (passing the return value of generative function application)
        to the next iteration for :code:`length` number of steps.
        The programmer must provide an initial value to start the chain of
        iterations off.

    Example
    -------

    .. jupyter-execute::

        import jax
        import genjax


        @genjax.gen
        def random_walk(key, prev):
            key, x = genjax.trace("x", genjax.Normal)(key, prev, 1.0)
            return (key, x)


        unfold = genjax.UnfoldCombinator(random_walk, 1000)
        init = 0.5
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(unfold))(key, (999, init))
        print(tr)
    """

    kernel: GenerativeFunction
    max_length: int

    def flatten(self):
        return (), (self.kernel, self.max_length)

    def get_trace_type(self, key, args, **kwargs):
        _ = args[0]
        args = args[1:]
        inner_type = self.kernel.get_trace_type(key, args, **kwargs)
        return VectorTraceType(inner_type, self.max_length)

    def _throw_bounds_host_exception(self, count: int):
        def _inner(count, transforms):
            raise Exception(
                f"\nUnfoldCombinator {self} received a length argument ({count}) longer than specified max length ({self.max_length})"
            )

        hcb.id_tap(
            lambda *args: _inner(*args),
            count,
            result=None,
        )
        return None

    def simulate(self, key, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # This inserts a host callback check for bounds checking.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length + 1),
            lambda *args: None,
        )

        def _inner(carry, x):
            count, key, state = carry
            key, tr = self.kernel.simulate(key, (state, *static_args))
            check = jnp.less(count, length + 1)
            state = concrete_cond(
                check,
                lambda *args: tr.get_retval(),
                lambda *args: state,
            )
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count = concrete_cond(
                check,
                lambda *args: count + 1,
                lambda *args: count,
            )
            return (count, key, state), (tr, index, state)

        (count, key, state), (tr, indices, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            indices,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        return key, unfold_tr

    # This checks the leaves of a choice map,
    # to determine if it is "out of bounds" for
    # the max static length of this combinator.
    def bounds_checker(self, v):
        lengths = []

        def _inner(v):
            if v.shape[-1] > self.max_length:
                raise Exception("Length of leaf longer than max length.")
            else:
                lengths.append(v.shape[-1])
                return v

        ret = jtu.tree_map(_inner, v)
        fixed_len = set(lengths)
        assert len(fixed_len) == 1
        return ret, fixed_len.pop()

    # This pads the leaves of a choice map up to
    # `self.max_length` -- so that we can scan
    # over the leading axes of the leaves.
    def padder(self, v):
        ndim = len(v.shape)
        pad_axes = list(
            (0, self.max_length - len(v)) if k == 0 else (0, 0)
            for k in range(0, ndim)
        )
        return (
            np.pad(v, pad_axes)
            if isinstance(v, np.ndarray)
            else jnp.pad(v, pad_axes)
        )

    def _importance_indexed(self, key, chm, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # Unwrap the index mask.
        inner_choice_map = chm.inner
        target_index = chm.get_index()

        def _inner(carry, slice):
            count, key, state = carry

            def _importance(key, state):
                return self.kernel.importance(
                    key, inner_choice_map, (state, *static_args)
                )

            def _simulate(key, state):
                key, tr = self.kernel.simulate(key, (state, *static_args))
                return key, (0.0, tr)

            check = count == target_index
            key, (w, tr) = concrete_cond(
                check,
                _importance,
                _simulate,
                key,
                state,
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, w = concrete_cond(
                check,
                lambda *args: (count + 1, tr.get_retval(), w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (w, tr, index, state)

        (count, key, state), (w, tr, indices, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            indices,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    def _importance_vcm(self, key, chm, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice

            def _importance(key, chm, state):
                return self.kernel.importance(key, chm, (state, *static_args))

            def _simulate(key, chm, state):
                key, tr = self.kernel.simulate(key, (state, *static_args))
                return key, (0.0, tr)

            check = count == chm.get_index()
            key, (w, tr) = concrete_cond(
                check,
                _importance,
                _simulate,
                key,
                chm,
                state,
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, w = concrete_cond(
                check,
                lambda *args: (count + 1, tr.get_retval(), w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (w, tr, index, state)

        (count, key, state), (w, tr, indices, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            indices,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    def _importance_fallback(self, key, chm, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # Check incoming choice map, and coerce to `VectorChoiceMap`
        # before passing into scan calls.
        chm, fixed_len = self.bounds_checker(chm)
        chm = jtu.tree_map(
            self.padder,
            chm,
        )
        if not isinstance(chm, VectorChoiceMap):
            indices = np.array(
                [
                    ind if ind < fixed_len else -1
                    for ind in range(0, self.max_length)
                ]
            )
            chm = VectorChoiceMap(
                indices,
                chm,
            )

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice

            def _importance(key, chm, state):
                return self.kernel.importance(key, chm, (state, *static_args))

            def _simulate(key, chm, state):
                key, tr = self.kernel.simulate(key, (state, *static_args))
                return key, (0.0, tr)

            check = count == chm.get_index()
            key, (w, tr) = concrete_cond(
                check,
                _importance,
                _simulate,
                key,
                chm,
                state,
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, weight = concrete_cond(
                check,
                lambda *args: (count + 1, tr.get_retval(), w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (w, tr, index, state)

        (count, key, state), (w, tr, indices, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            indices,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    def importance(self, key, chm, args):
        length = args[0]

        # This inserts a host callback check for bounds checking.
        # At runtime, if the bounds are exceeded -- an error
        # will be emitted.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length + 1),
            lambda *args: None,
        )

        if isinstance(chm, VectorChoiceMap):
            return self._importance_vcm(key, chm, args)
        else:
            return self._importance_fallback(key, chm, args)

    # The choice map has an index mask, can efficiently
    # update.
    def _update_indexed(self, key, prev, chm, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # The purpose of this branch is to efficiently perform single
        # index updates. This is a common pattern in e.g. SMC, so we
        # optimize for it here.

        # Unwrap the index mask.
        inner_choice_map = chm.inner
        target_index = chm.get_index()

        def _inner(carry, slice):
            count, key, state = carry
            (prev,) = slice

            def _update(key, prev, state):
                return self.kernel.update(
                    key,
                    prev,
                    inner_choice_map,
                    (state, *static_args),
                )

            def _fallthrough(key, prev, state):
                return self.kernel.update(
                    key,
                    prev,
                    EmptyChoiceMap(),
                    (state, *static_args),
                )

            # Here, we check the index.
            check = count == target_index
            key, (retdiff, w, tr, discard) = concrete_cond(
                check,
                _update,
                _fallthrough,
                key,
                prev,
                state,
            )

            # `Unfold` has upper-bound allocation size,
            # but any particular invocation may go less than
            # that size -- here, we fill define fallbacks to
            # fill up to the allocation size.
            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, weight = concrete_cond(
                check,
                lambda *args: (count + 1, retdiff, w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (state, w, tr, d, index)

        (count, key, state), (retdiff, w, tr, d, indices) = jax.lax.scan(
            _inner,
            (0, key, state),
            (prev,),
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            indices,
            tr,
            args,
            retdiff.get_val(),
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (retdiff, w, unfold_tr, d)

    # The choice map is a vector choice map.
    def _update_vcm(self, key, prev, chm, diffs):
        length = diffs[0]
        state = diffs[1]
        static_args = diffs[2:]
        args = tuple(map(lambda v: v.get_val(), diffs))

        # Here, we skip any choice map pre-setup -
        # assuming the user is encoding information directly
        # so `chm: VectorChoiceMap`.

        # The scan call here is the same as the fallback call in
        # `_update_fallback`.
        def _inner(carry, slice):
            count, key, state = carry
            (prev, chm) = slice

            def _update(key, prev, chm, state):
                return self.kernel.update(
                    key, prev, chm, (state, *static_args)
                )

            def _fallthrough(key, prev, chm, state):
                return self.kernel.update(
                    key, prev, EmptyChoiceMap(), (state, *static_args)
                )

            check = count == chm.get_index()
            key, (retdiff, w, tr, d) = concrete_cond(
                check, _update, _fallthrough, key, prev, chm, state
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, weight = concrete_cond(
                check,
                lambda *args: (count + 1, retdiff, w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (state, w, tr, d, index)

        (count, key, state), (retdiff, w, tr, d, indices) = jax.lax.scan(
            _inner,
            (0, key, state),
            (prev, chm),
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            indices,
            tr,
            args,
            retdiff.get_val(),
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (retdiff, w, unfold_tr, d)

    # The choice map doesn't carry optimization info.
    def _update_fallback(self, key, prev, chm, diffs):
        length = diffs[0]
        state = diffs[1]
        static_args = diffs[2:]
        args = tuple(map(lambda v: v.get_val(), diffs))

        # Check incoming choice map, and coerce to `VectorChoiceMap`
        # before passing into scan calls.
        self.bounds_checker(chm)
        chm = jtu.tree_map(
            self.padder,
            chm,
        )
        chm = VectorChoiceMap(
            np.array([ind for ind in range(0, self.max_length)]),
            chm,
        )

        # The actual semantics of update are carried out by a scan
        # call.

        def _inner(carry, slice):
            count, key, state = carry
            (prev, chm) = slice

            def _update(key, prev, chm, state):
                return self.kernel.update(
                    key, prev, chm, (state, *static_args)
                )

            def _fallthrough(key, prev, chm, state):
                return self.kernel.update(
                    key, prev, EmptyChoiceMap(), (state, *static_args)
                )

            check = count == chm.get_index()
            key, (retdiff, w, tr, d) = concrete_cond(
                check, _update, _fallthrough, key, prev, chm, state
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, weight = concrete_cond(
                check,
                lambda *args: (count + 1, retdiff, w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (state, w, tr, d, index)

        (_, key, _), (retdiff, w, tr, d, indices) = jax.lax.scan(
            _inner,
            (0, key, state),
            (prev, chm),
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            indices,
            tr,
            args,
            retdiff.get_val(),
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (retdiff, w, unfold_tr, d)

    def update(self, key, prev, chm, diffs):
        length = diffs[0].get_val()

        # Unwrap the previous trace at this address
        # we should get a `VectorChoiceMap`.
        # We don't need the index indicators from the trace,
        # so we can just unwrap it.
        assert isinstance(prev, UnfoldTrace)
        prev = prev.inner

        # This inserts a host callback check for bounds checking.
        # If we go out of bounds on device, it throws to the
        # Python runtime -- which will raise.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length + 1),
            lambda *args: None,
        )

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
                f"\nUnfoldCombinator {self} received a choice map with mismatched indices (at index {index}) in assess."
            )

        hcb.id_tap(
            lambda *args: _inner(*args),
            index,
            result=None,
        )
        return None

    def assess(self, key, chm, args):
        assert isinstance(chm, VectorChoiceMap)
        length = args[0]

        # This inserts a host callback check for bounds checking.
        # At runtime, if the bounds are exceeded -- an error
        # will be emitted.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length + 1),
            lambda *args: None,
        )

        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice

            check = count == chm.get_index()

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

            key, (retval, score) = self.kernel.assess(
                key, chm, (state, *static_args)
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, score = concrete_cond(
                check,
                lambda *args: (count + 1, retval, score),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (state, score, index)

        (_, key, state), (retval, score, _) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        score = jnp.sum(score)
        return key, (retval, score)
