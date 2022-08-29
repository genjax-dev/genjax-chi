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
Supports zero-cost effect handling implementations of each of the
generative function interfaces -- dispatching on `trace_p`
and `extern_p` primitives for each of the GFI methods.

These handlers build on top of the CPS/effect handling interpreter
in `genjax.core`.
"""

import jax
import jax.tree_util as jtu
from genjax.core import Handler, handle
from .trie import Trie
from .intrinsics import gen_fn_p

#####
# GFI handlers
#####

# Note: these handlers _do not manipulate runtime values_ --
# the pointers they hold to objects like `v` and `score` are JAX `Tracer`
# values. When we do computations with these values,
# it adds to the `Jaxpr` trace.
#
# So the trick is write `callable` to coerce the return of the `Jaxpr`
# to send out the accumulated state we want.


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
    def trace(self, f, key, *args, addr, gen_fn, **kwargs):
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
    def __init__(self, constraints):
        self.handles = [
            gen_fn_p,
        ]
        self.state = Trie({})
        self.level = []
        self.score = 0.0
        self.weight = 0.0
        self.constraints = constraints
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, **kwargs):
        if self.constraints.has_choice(addr):
            chm = self.constraints.get_choice(addr)
            key, (w, tr) = gen_fn.importance(key, chm, args)
            self.state[addr] = tr
            self.score += tr.get_score()
            self.weight += w
            v = tr.get_retval()
        else:
            key, tr = gen_fn.simulate(key, args)
            self.state[addr] = tr
            self.score += tr.get_score()
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


class Diff(Handler):
    def __init__(self, original, new):
        self.handles = [
            gen_fn_p,
        ]
        self.level = []
        self.weight = 0.0
        self.original = original.get_choices()
        self.choice_change = new
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, **kwargs):
        has_previous = self.original.has_choice(addr)
        constrained = self.choice_change.has_choice(addr)

        if has_previous:
            prev_tr = self.original.get_choice(addr)

        if constrained:
            chm = self.choice_change.get_choice(addr)

        if has_previous and constrained:
            key, (w, v) = gen_fn.diff(key, prev_tr, chm, args)
            self.weight += w
        elif has_previous:
            v = prev_tr.get_retval()
        elif constrained:
            key, (w, tr) = gen_fn.importance(key, chm, args)
            self.weight += w
            v = tr.get_retval()

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, (self.weight, ret)

    # Transform a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _transform(self, f, *args):
        expr = jax.make_jaxpr(f)(*args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class Update(Handler):
    def __init__(self, original, new):
        self.handles = [
            gen_fn_p,
        ]
        self.level = []
        self.state = Trie({})
        self.discard = Trie({})
        self.weight = 0.0
        self.original = original.get_choices()
        self.choice_change = new
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, **kwargs):
        has_previous = self.original.has_choice(addr)
        constrained = self.choice_change.has_choice(addr)
        if has_previous:
            prev_tr = self.original.get_choice(addr)
        if constrained:
            chm = self.choice_change.get_choice(addr)
        if has_previous and constrained:
            key, (w, tr, discard) = gen_fn.update(key, prev_tr, chm, args)
            self.weight += w
            self.state[addr] = tr
            self.discard[addr] = discard
            v = tr.get_retval()
        elif has_previous:
            self.state[addr] = prev_tr
            v = prev_tr.get_retval()
        elif constrained:
            key, (w, tr) = gen_fn.importance(key, chm, args)
            v = tr.get_retval()
            self.state[addr] = tr
            self.weight += w

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, (self.weight, ret, self.state, self.discard)

    def _transform(self, f, *args):
        expr = jax.make_jaxpr(f)(*args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class ArgumentGradients(Handler):
    def __init__(self, tr, argnums):
        self.handles = [
            gen_fn_p,
        ]
        self.argnums = argnums
        self.level = []
        self.score = 0.0
        self.source = tr.get_choices()
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, **kwargs):
        if self.source.has_choice(addr):
            sub_tr = self.source.get_choice(addr)
            chm = sub_tr.get_choices()
            key, (w, tr) = gen_fn.importance(key, chm, args)
            v = tr.get_retval()
            self.score += w
        else:
            key, tr = gen_fn.simulate(key, args)
            v = tr.get_retval()

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *_ = f(key, *v)
            return self.score, key

    def _transform(self, f, *args):
        expr = jax.make_jaxpr(f)(*args)
        fn = handle([self], expr)
        fn = jax.grad(fn, self.argnums, has_aux=True)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class ChoiceGradients(Handler):
    def __init__(self, tr):
        self.tr = tr

    def _transform(self, f, key, *args):
        def diff(key, chm):
            handler = Diff(self.tr, chm)
            expr = jax.make_jaxpr(f)(key, *args)
            fn = handle([handler], expr)
            key, (w, _) = fn(key, *args)
            return self.tr.get_score() + w, key

        gradded = jax.grad(diff, argnums=1, has_aux=True)
        return gradded

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
    def _inner(key, chm, args):
        fn = Importance(chm).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, r, chm, score) = fn(key, *in_args)
        return key, (w, (f, args, tuple(r), chm, score))

    return lambda key, chm, args: _inner(key, chm, args)


def diff(f):
    def _inner(key, prev, new, args):
        fn = Diff(prev, new).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, ret) = fn(key, *in_args)
        return key, (w, ret)

    return lambda key, prev, new, args: _inner(key, prev, new, args)


def update(f):
    def _inner(key, prev, new, args):
        fn = Update(prev, new).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, ret, chm, discard) = fn(key, *in_args)
        return key, (
            w,
            (f, args, tuple(ret), chm, prev.get_score() + w),
            discard,
        )

    return lambda key, prev, new, args: _inner(key, prev, new, args)


def arg_grad(f, argnums):
    def _inner(key, tr, args):
        fn = ArgumentGradients(tr, argnums).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        arg_grads, key = fn(key, *in_args)
        return key, arg_grads

    return lambda key, tr, args: _inner(key, tr, args)


def choice_grad(f):
    def _inner(key, tr, chm, args):
        fn = ChoiceGradients(tr).transform(f)(key, *args)
        choice_grads, key = fn(key, chm)
        return key, choice_grads

    return lambda key, tr, chm, args: _inner(key, tr, chm, args)
