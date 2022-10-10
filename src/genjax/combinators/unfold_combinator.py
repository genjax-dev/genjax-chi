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
as kernels (a kernel generative function can accept
their previous output as input).
"""

from dataclasses import dataclass
from typing import Any
from typing import Sequence
from typing import Tuple

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax.combinators.combinator_datatypes import VectorChoiceMap
from genjax.combinators.combinator_tracetypes import VectorTraceType
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Trace
from genjax.core.masks import BooleanMask
from genjax.core.specialization import concrete_cond


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
            key, x = genjax.trace("x", genjax.Normal)(key, (prev, 1.0))
            return (key, x)


        unfold = genjax.UnfoldCombinator(random_walk, 1000)
        init = 0.5
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(unfold))(key, (1000, init,))
        print(tr)
    """

    kernel: GenerativeFunction
    max_length: int

    def get_trace_type(self, key, args, **kwargs):
        _ = args[0]
        args = args[1:]
        inner_type = self.kernel.get_trace_type(key, args, **kwargs)
        return VectorTraceType(inner_type, self.max_length)

    def flatten(self):
        return (), (self.kernel, self.max_length)

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

    def __call__(self, key, *args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # This inserts a host callback check for bounds checking.
        check = jnp.less(self.max_length, length)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length),
            lambda *args: None,
        )

        def _inner(carry, x):
            key, state = carry
            key, tr = self.kernel.simulate(key, (state, *static_args))
            return (key, *tr.get_retval()), ()

        (key, retval), _ = jax.lax.scan(
            _inner,
            (key, state),
            None,
            length=self.max_length,
        )
        return key, retval

    def simulate(self, key, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # This inserts a host callback check for bounds checking.
        check = jnp.less(self.max_length, length)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length),
            lambda *args: None,
        )

        def _inner(carry, x):
            count, key, state = carry
            key, tr = self.kernel.simulate(key, (state, *static_args))
            check = jnp.less(count, length)
            retval = concrete_cond(
                check,
                lambda *args: tr.get_retval(),
                lambda *args: (state,),
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
            return (count, key, *retval), (tr, index)

        (count, key, retval), (tr, indices) = jax.lax.scan(
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
            (retval,),
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

    @BooleanMask.collapse_boundary
    def importance(self, key, chm, args):
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
            chm = VectorChoiceMap.new(
                indices,
                chm,
            )

        # This inserts a host callback check for bounds checking.
        # At runtime, if the bounds are exceeded -- an error
        # will be emitted.
        check = jnp.less(self.max_length, length)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length),
            lambda *args: None,
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

            check = jnp.less(count, length)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, retval, weight = concrete_cond(
                check,
                lambda *args: (count + 1, *tr.get_retval(), w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, retval), (w, tr, index)

        (count, key, retval), (w, tr, indices) = jax.lax.scan(
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
            (retval,),
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    @BooleanMask.collapse_boundary
    def update(self, key, prev, chm, args):
        assert isinstance(prev, UnfoldTrace)
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # Unwrap the previous trace at this address
        # we should get a `VectorChoiceMap`.
        # We don't need the index indicators, so we can just
        # unwrap it.
        prev = prev.get_choices()
        assert isinstance(prev, VectorChoiceMap)
        prev = prev.inner

        # Check incoming choice map, and coerce to `VectorChoiceMap`
        # before passing into scan calls.
        self.bounds_checker(chm)
        chm = jtu.tree_map(
            self.padder,
            chm,
        )
        if not isinstance(chm, VectorChoiceMap):
            chm = VectorChoiceMap.new(
                np.array([ind for ind in range(0, self.max_length)]),
                chm,
            )

        # This inserts a host callback check for bounds checking.
        check = jnp.less(self.max_length, length)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length),
            lambda *args: None,
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
            key, (w, tr, d) = concrete_cond(
                check, _update, _fallthrough, key, prev, chm, state
            )

            check = jnp.less(count, length)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, weight = concrete_cond(
                check,
                lambda *args: (count + 1, *tr.get_retval(), w),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (w, tr, d, index)

        (count, key, retval), (w, tr, d, indices) = jax.lax.scan(
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
            (retval,),
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr, d)
