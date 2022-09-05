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
    Mask,
)
from genjax.builtin.shape_analysis import choice_map_shape
from dataclasses import dataclass
from typing import Any, Tuple
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
        return VectorChoiceMap(self.subtrace, self.length)

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
        key, tr = jax.jit(genjax.simulate(unfold))(key, (init,))
        print(tr)
    """

    kernel: GenerativeFunction
    length: int

    def flatten(self):
        return (self.length,), (self.kernel,)

    @classmethod
    def unflatten(cls, data, xs):
        return UnfoldCombinator(*data, *xs)

    def __call__(self, key, *args):
        def _inner(carry, x):
            key, args = carry
            key, tr = self.kernel.simulate(key, args)
            return (key, tr.get_retval()), ()

        return jax.lax.scan(
            _inner,
            (key, args),
            None,
            length=self.length,
        )

    def simulate(self, key, args):
        def _inner(carry, x):
            key, tr = self.kernel.simulate(*carry)
            retval = tr.get_retval()
            return (key, retval), tr

        (key, retval), tr = jax.lax.scan(
            _inner,
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

        # This allows the user to specialize if they pass in a VectorChoiceMap,
        # versus fallback to generic Mask lookups.
        # Similar to update below, this section provides some control
        # to the user in terms of how importance is specialized
        # onto the input choice map.

        length = self.length
        assert length > 0
        if not isinstance(chm, VectorChoiceMap) and not isinstance(chm, Mask):
            _, treedef, shape = choice_map_shape(self.kernel)(key, args)
            chm_vectored, mask_vectored = prepare_vectorized_choice_map(
                shape, treedef, length, chm
            )

            chm = Mask(chm_vectored, mask_vectored)
        if isinstance(chm, VectorChoiceMap):
            chm = chm.subtrace

        # The actual semantics of importance are carried out by a scan
        # call.

        def _inner(carry, slice):
            key, args = carry
            key, (w, tr) = self.kernel.importance(key, slice, args)
            retval = tr.get_retval()
            return (key, retval), (w, tr)

        (key, retval), (w, tr) = jax.lax.scan(
            _inner,
            (key, args),
            chm,
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

    def update(self, key, prev, chm, args):

        # This section is sort of like optimization / pre-setup.
        # There's a lot of room for improvement, especially in
        # asymptotics.

        length = self.length
        assert length > 0
        if not isinstance(chm, VectorChoiceMap) and not isinstance(chm, Mask):
            _, treedef, shape = choice_map_shape(self.kernel)(key, args)
            chm_vectored, mask_vectored = prepare_vectorized_choice_map(
                shape, treedef, length, chm
            )

            chm = Mask(chm_vectored, mask_vectored)
        if isinstance(chm, VectorChoiceMap):
            chm = chm.subtrace

        prev = prev.get_choices()
        assert isinstance(prev, VectorChoiceMap)
        prev = prev.subtrace

        # The actual semantics of update are carried out by a scan
        # call.

        def _inner(carry, slice):
            key, args = carry
            prev, new = slice
            key, (w, tr, d) = self.kernel.update(key, prev, new, args)
            retval = tr.get_retval()
            return (key, retval), (w, tr, d)

        (key, retval), (w, tr, d) = jax.lax.scan(
            _inner,
            (key, args),
            (prev, chm),
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
        return key, (w, unfold_tr, d)
