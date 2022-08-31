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
This module holds an implementation of :code:`trace` handlers, modified
to support dynamic lookups of addresses in hierarchical choice maps from
the builtin language.

The handler implementations below are similar to those found in
:code:`genjax.builtin.handlers`, but these implementations
explicitly include :code:`jax.lax.cond` calls to switch
on dynamic lookups.
"""

import jax
import jax.tree_util as jtu
from genjax.core import Handler, handle
from genjax.builtin.trie import Trie
from genjax.builtin.intrinsics import gen_fn_p
import genjax.interface as gfi

#####
# Dynamic GFI handlers
#####


class Sample(Handler):
    def __init__(self):
        self.handles = [
            gen_fn_p,
        ]
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, **kwargs):
        key, tr = gen_fn.simulate(key, args)
        v = tr.get_retval()

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, ret

    # Transform a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _transform(self, f, *args):
        expr = jax.make_jaxpr(f)(*args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class Simulate(Handler):
    def __init__(self):
        self.handles = [
            gen_fn_p,
        ]
        self.state = Trie({})
        self.level = []
        self.score = 0.0
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, mask, *args, addr, gen_fn, **kwargs):
        key, tr = gen_fn.simulate(key, args)
        score = tr.get_score()
        v = tr.get_retval()
        self.state[addr] = tr
        self.score += score

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, (ret, self.state, self.score)

    # Transform a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _transform(self, f, *args):
        expr = jax.make_jaxpr(f)(*args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class Importance(Handler):
    def __init__(self, mask, constraints):
        self.handles = [
            gen_fn_p,
        ]
        self.state = Trie({})
        self.level = []
        self.score = 0.0
        self.weight = 0.0
        self.mask = mask
        self.constraints = constraints
        self.return_or_continue = False

    def _simulate_branch(self, gen_fn, key, chm, args):
        key, tr = gfi.simulate(gen_fn)(key, args)
        return key, (0.0, tr)

    def _importance_branch(self, gen_fn, key, chm, args):
        key, (w, tr) = gfi.importance(gen_fn)(key, chm, args)
        return key, (w, tr)

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, **kwargs):
        mask_check = self.mask.get_choice(addr)
        submap = self.constraints.get_choice(addr)
        key, (w, tr) = jax.lax.cond(
            mask_check,
            lambda key, chm, args: self._importance_branch(
                gen_fn, key, submap, args
            ),
            lambda key, chm, args: self._simulate_branch(
                gen_fn, key, submap, args
            ),
            key,
            self.constraints,
            args,
        )
        self.state[addr] = tr
        self.score += tr.get_score()
        self.weight += w
        v = tr.get_retval()

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, (self.weight, ret, self.state, self.score)

    # Transform a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _transform(self, f, *args):
        expr = jax.make_jaxpr(f)(*args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


#####
# Generative function interface
#####


def sample(f):
    def _inner(key, args):
        fn = Sample().transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, v = fn(key, *in_args)
        return key, v

    return lambda key, args: _inner(key, args)


def simulate(f):
    def _inner(key, args):
        fn = Simulate().transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (r, chm, score) = fn(key, *in_args)
        return key, (f, args, tuple(r), chm, score)

    return lambda key, args: _inner(key, args)


def importance(f):
    def _inner(key, mask, chm, args):
        fn = Importance(mask, chm).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, r, chm, score) = fn(key, *in_args)
        return key, (w, (f, args, tuple(r), chm, score))

    return lambda key, mask, chm, args: _inner(key, mask, chm, args)
