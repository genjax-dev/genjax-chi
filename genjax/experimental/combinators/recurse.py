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
This module holds an experimental (🔪) combinator which allows GFI methods
to correctly record generative trace on tail recursive code 
(the size of the resulting choice map may be infinite).

To implement this functionality, we utilize a new experimental module
in JAX called `host_callback`_ which allows the XLA runtime to call back
into Python runtime (including JIT-compiled XLA code!).

As part of the exposed constructor for the :code:`RecurseCombinator`,
the programmer is allowed to specify an initial shape bound as well the
growing interval (if the underlying XLA :code:`while_loop` exceeds the
initial shape bound).

.. _host_callback: https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import genjax.interface as gfi
from genjax.core.datatypes import (
    GenerativeFunction,
    Trace,
)
from genjax.builtin.shape_analysis import trace_shape
from dataclasses import dataclass
from typing import Any, Tuple, Callable

#####
# RecurseTrace
#####


@dataclass
class RecurseTrace(Trace):
    gen_fn: GenerativeFunction
    inner: Trace
    iter: int
    args: Tuple
    retval: Tuple
    score: Any

    def flatten(self):
        return (self.inner, self.iter, self.args, self.retval, self.score), (
            self.gen_fn,
        )

    @classmethod
    def unflatten(cls, xs, data):
        return RecurseTrace(*xs, *data)

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def get_choices(self):
        self.inner.get_choices()

    def get_gen_fn(self):
        return self.gen_fn


#####
# RecurseCombinator
#####


@dataclass
class RecurseCombinator(GenerativeFunction):
    cond_fn: Callable
    while_gen_fn: GenerativeFunction
    shape_bound: int
    growth_interval: int

    def __call__(self, key, *args):
        return jax.lax.while_loop(
            lambda args: self.cond_fn(*args),
            lambda args: self.while_gen_fn(*args),
            (key, args),
        )

    def flatten(self):
        return (), (
            self.cond_fn,
            self.while_gen_fn,
            self.shape_bound,
            self.growth_interval,
        )

    @classmethod
    def unflatten(cls, xs, data):
        return RecurseCombinator(*xs, *data)

    def simulate(self, key, args):
        def _check(key, tr, count, *args):
            keep_going = self.cond_fn(*args)
            bound_check = count <= self.shape_bound
            return keep_going and bound_check

        def _inner(key, tr, count, *args):
            key, iter = gfi.simulate(self.while_gen_fn)(key, args)
            score = tr.get_score() + iter.get_score()
            retval = iter.get_retval()
            tr = RecurseTrace(
                self,
                iter,
                count,
                args,
                iter.get_retval(),
                jnp.sum(score),
            )
            return (key, tr, count + 1, *retval)

        _, _, blank_tr = trace_shape(self.while_gen_fn)(key, args)
        blank_tr = jtu.tree_map(lambda v: jnp.zeros(v.shape, v.dtype), blank_tr)
        input_tr = RecurseTrace(
            self,
            blank_tr,
            0,
            args,
            blank_tr.get_retval(),
            blank_tr.get_score(),
        )

        key, tr, count, *retval = jax.lax.while_loop(
            lambda args: _check(*args),
            lambda args: _inner(*args),
            (key, input_tr, 0, *args),
        )

        return key, tr
