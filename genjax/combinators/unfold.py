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

import jax
import jax.numpy as jnp
from genjax.core.datatypes import (
    GenerativeFunction,
    Trace,
    EmptyChoiceMap,
    IndexMask,
)
from genjax.core.tracetypes import TraceType
import jax.experimental.host_callback as hcb
from genjax.core.specialization import concrete_cond
from dataclasses import dataclass
from genjax.combinators.vector_choice_map import VectorChoiceMap
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp
from typing import Any, Tuple, Sequence

#####
# UnfoldTrace
#####


@dataclass
class UnfoldTrace(Trace):
    gen_fn: GenerativeFunction
    length: int
    mask: Sequence
    subtrace: Trace
    args: Tuple
    retval: Any
    score: jnp.float32

    def get_args(self):
        return self.args

    def get_choices(self):
        return VectorChoiceMap(self.subtrace)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def flatten(self):
        return (
            self.length,
            self.mask,
            self.subtrace,
            self.args,
            self.retval,
            self.score,
        ), (self.gen_fn,)

    @classmethod
    def unflatten(cls, data, xs):
        return UnfoldTrace(*data, *xs)


#####
# ListTraceType
#####


@dataclass
class ListTraceType(TraceType):
    inner: TraceType
    length: int

    def flatten(self):
        return (), (self.inner, self.length)

    @classmethod
    def unflatten(cls, xs, data):
        return ListTraceType(*xs, *data)

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"[{self.length}; "),
                gpp._nest(indent, gpp._pformat(self.inner, **kwargs)),
                pp.brk(),
                pp.text("]"),
                pp.brk(),
                pp.text("return_type -> "),
                gpp._pformat(self.inner.get_rettype(), **kwargs),
            ]
        )

    def __subseteq__(self, other):
        return False

    def get_rettype(self):
        return self.inner.get_rettype()


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
        return ListTraceType(inner_type, self.max_length)

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

    @classmethod
    def unflatten(cls, data, xs):
        return UnfoldCombinator(*data, *xs)

    def __call__(self, key, *args):
        state = args[1]
        static_args = args[2:]

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
                lambda *args: (state,),
                lambda *args: tr.get_retval(),
            )
            count = concrete_cond(
                check,
                lambda *args: count + 1,
                lambda *args: count,
            )
            return (count, key, *retval), (tr, check)

        (count, key, retval), (tr, mask) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            count,
            mask,
            tr,
            args,
            (retval,),
            jnp.sum(tr.get_score()),
        )

        return key, unfold_tr

    def importance(self, key, chm, args):

        length = args[0]
        args = args[1:]

        # This inserts a host callback check for bounds checking.
        check = jnp.less(self.max_length, length)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length),
            lambda *args: None,
        )

        def _inner(carry, slice):
            count, key, retval = carry
            if isinstance(chm, IndexMask):
                check = count == chm.get_index()
            else:
                check = True

                def _importance(key, chm, retval):
                    return self.kernel.importance(key, chm, retval)

                def _simulate(key, chm, retval):
                    key, tr = self.kernel.simulate(key, retval)
                    return key, (0.0, tr)

                key, (w, tr) = concrete_cond(
                    check, _importance, _simulate, key, chm, retval
                )

                check = jnp.less(count, length)
                count, retval, weight = concrete_cond(
                    check,
                    lambda *args: (count + 1, tr.get_retval(), w),
                    lambda *args: (count, retval, 0.0),
                )
                return (count, key, retval), (w, tr, check)

        (count, key, retval), (w, tr, mask) = jax.lax.scan(
            _inner,
            (0, key, args),
            None,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            count,
            mask,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    def update(self, key, prev, chm, args):

        length = args[0]
        args = args[1:]

        # This inserts a host callback check for bounds checking.
        check = jnp.less(self.max_length, length)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length),
            lambda *args: None,
        )

        prev = prev.get_choices()
        assert isinstance(prev, VectorChoiceMap)
        prev = prev.subtrace

        # The actual semantics of update are carried out by a scan
        # call.

        def _inner(carry, slice):
            count, key, retval = carry
            prev = slice

            if isinstance(chm, IndexMask):
                check = count == chm.get_index()
            else:
                check = False

            def _update(key, prev, chm, retval):
                return self.kernel.update(key, prev, chm, retval)

            def _fallthrough(key, prev, chm, retval):
                return self.kernel.update(key, prev, EmptyChoiceMap(), retval)

            key, (w, tr, d) = concrete_cond(
                check, _update, _fallthrough, key, prev, chm, retval
            )

            check = jnp.less(count, length)
            count, retval, weight = concrete_cond(
                check,
                lambda *args: (count + 1, tr.get_retval(), w),
                lambda *args: (count, retval, 0.0),
            )
            return (count, key, retval), (w, tr, d, check)

        (count, key, retval), (w, tr, d, mask) = jax.lax.scan(
            _inner,
            (0, key, args),
            prev,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            count,
            mask,
            tr,
            args,
            retval,
            jnp.sum(tr.get_score()),
        )

        w = jnp.sum(w)
        return key, (w, unfold_tr, d)
