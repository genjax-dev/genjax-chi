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
from .core import Handler, handle, lift
from .intrinsics import (
    primitive_p,
    trace_p,
    batched_trace_p,
    splice_p,
    unsplice_p,
)
from genjax.datatypes import ChoiceMap
from genjax.datatypes import Trace
from genjax import PrimitiveGenerativeFunction

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


class Simulate(Handler):
    def __init__(self):
        self.handles = [
            primitive_p,
            trace_p,
            batched_trace_p,
            splice_p,
            unsplice_p,
        ]
        self.state = ChoiceMap([])
        self.level = []
        self.score = 0.0
        self.return_or_continue = False

    # Handle primitive sites -- deferring `simulate`
    # to a custom implementation.
    def primitive(self, f, key, *args, addr, prim, **kwargs):
        full = ".".join((*self.level, addr))
        key, tr = prim.simulate(key, args)
        score = tr.get_score()
        v = tr.get_retval()
        self.state[full] = tr
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (ret, self.state, self.score)

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, call, **kwargs):
        full = ".".join((*self.level, addr))
        key, tr = simulate(call)(key, args)
        score = tr.get_score()
        v = tr.get_retval()
        self.state[full] = tr
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (ret, self.state, self.score)

    # Handle batched_trace sites -- perform codegen onto the `Jaxpr` trace.
    def batched_trace(self, f, *args, addr, prim, **kwargs):
        prim = prim()
        key, v = jax.vmap(prim.sample)(*args)
        full = ".".join((*self.level, addr))
        score = prim.score(v, *args[1:])
        self.state[full] = (v, score)
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (ret, self.state, self.score)

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f(())

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f(())
        except BaseException:
            return f(())

    # Transform a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _transform(self, f, *args):
        expr = lift(f, *args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class Importance(Handler):
    def __init__(self, choice_map):
        self.handles = [
            trace_p,
            primitive_p,
            batched_trace_p,
            splice_p,
            unsplice_p,
        ]
        self.state = ChoiceMap([])
        self.level = []
        self.score = 0.0
        self.weight = 0.0
        self.obs = choice_map
        self.return_or_continue = False

    # Handle primitive sites -- deferring `importance`
    # to a custom implementation.
    def primitive(self, f, key, *args, addr, prim, **kwargs):
        full = ".".join((*self.level, addr))
        if self.obs.has_leaf(full):
            chm = self.obs.get_leaf(full)
            key, (w, tr) = prim.importance(key, chm, args)
            self.state[full] = tr
            self.score += tr.get_score()
            self.weight += w
            v = tr.get_retval()
        else:
            key, tr = prim.simulate(key, args)
            self.state[full] = tr
            self.score += tr.get_score()
            v = tr.get_retval()

        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (self.weight, ret, self.state, self.score)

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, prim, **kwargs):
        full = ".".join((*self.level, addr))
        if self.obs.has_submap(full):
            chm = self.obs.get_submap(full)
            key, (w, tr) = importance(f)(key, chm, args)
            self.state[full] = (v, score)
            self.score += tr.get_score()
            self.weight += w
            v = tr.get_retval()
        else:
            key, tr = simulate(f)(key, args)
            self.state[full] = tr
            self.score += tr.get_score()

        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (self.weight, ret, self.state, self.score)

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def batched_trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        full = ".".join((*self.level, addr))
        if self.obs.has_choice(full):
            v = self.obs.get_value(full)
            score = prim.score(v, *args[1:])
            self.state[full] = (v, score)
            self.weight += score
        else:
            key, v = jax.vmap(prim.sample)(*args, **kwargs)
            score = prim.score(v, *args[1:])
            self.state[full] = (v, score)
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (self.weight, ret, self.state, self.score)

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f(())

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f(())
        except BaseException:
            return f(())

    # Transform a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _transform(self, f, *args):
        expr = lift(f, *args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class Diff(Handler):
    def __init__(self, original, new):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.level = []
        self.weight = 0.0
        self.original = original.get_choices()
        self.choice_change = new
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        full = ".".join((*self.level, addr))
        has_previous = self.original.has_choice(full)
        constrained = self.choice_change.has_choice(full)

        if has_previous:
            prev_choice = self.original.get_value(full)
            prev_score = self.original.get_score(full)

        if constrained:
            v = self.choice_change.get_value(full)
        elif has_previous:
            v = prev_choice
        else:
            key, v = prim.sample(key, *args[1:])
        score = prim.score(v, *args[1:])

        if has_previous:
            self.weight += score - prev_score
        elif constrained:
            self.weight += score

        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (self.weight, ret)

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f(())

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f(())
        except BaseException:
            return f(())

    # Transform a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _transform(self, f, *args):
        expr = lift(f, *args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class Update(Handler):
    def __init__(self, original, new):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.level = []
        self.state = ChoiceMap([])
        self.weight = 0.0
        self.original = original.get_choices()
        self.choice_change = new
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        full = ".".join((*self.level, addr))
        if self.choice_change.has_choice(full):
            v = self.choice_change.get_value(full)
            forward = prim.score(v, *args[1:])
            backward = self.original.get_score(full)
            self.state[full] = (v, forward)
            self.weight += forward - backward
        else:
            v = self.original.get_value(full)
            self.state[full] = (v, self.original.get_score(full))
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, v)
            return key, (self.weight, ret, self.state)

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f(())

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f(())
        except BaseException:
            return f(())

    # Transform a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _transform(self, f, *args):
        expr = lift(f, *args)
        fn = handle([self], expr)
        return fn

    def transform(self, f):
        return lambda *args: self._transform(f, *args)


class ArgumentGradients(Handler):
    def __init__(self, tr, argnums):
        self.handles = [primitive_p, trace_p, splice_p, unsplice_p]
        self.argnums = argnums
        self.level = []
        self.score = 0.0
        self.tr = tr
        self.sources = tr.get_choices()
        self.return_or_continue = False

    # Handle primitive sites -- perform codegen onto the `Jaxpr` trace.
    def primitive(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        full = ".".join((*self.level, addr))
        chm = self.sources.get_submap(full)
        key, (w, tr) = prim.importance(key, chm, *args[1:])
        v = tr.get_retval()
        self.score += w
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *_ = f(key, v)
            return self.score, key

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        full = ".".join((*self.level, addr))
        v = self.sources.get_value(full)
        score = prim.score(v, *args[1:])
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, *_ = f(key, v)
            return self.score, key

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f(())

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f(())
        except BaseException:
            return f(())

    def _transform(self, f, *args):
        expr = lift(f, *args)
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
            expr = lift(f, key, *args)
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
    def _inner(key, *args):
        fn = Sample().transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        k, *v = fn(key, *in_args)
        return k, (*v,)

    return lambda key, *args: _inner(key, *args)


def simulate(f):
    def _inner(key, args):
        fn = Simulate().transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (r, chm, score) = fn(key, *in_args)
        return key, Trace(f, args, tuple(r), chm, score)

    # If `f` is an primitive generative function, pass
    # the call to the method.
    if isinstance(f, PrimitiveGenerativeFunction):
        return lambda key, args: f.simulate(key, args)
    else:
        return lambda key, args: _inner(key, args)


def importance(f):
    def _inner(key, chm, args):
        fn = Importance(chm).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, r, chm, score) = fn(key, *in_args)
        return key, (w, Trace(f, args, tuple(r), chm, score))

    # If `f` is an primitive generative function, pass
    # the call to the method.
    if isinstance(f, PrimitiveGenerativeFunction):
        return lambda key, chm, args: f.importance(key, chm, args)
    else:
        return lambda key, chm, args: _inner(key, chm, args)


def diff(f):
    def _inner(key, original, new, args):
        fn = Diff(original, new).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, ret) = fn(key, *in_args)
        return key, (w, ret)

    return lambda key, original, new, args: _inner(key, original, new, args)


def update(f):
    def _inner(key, original, new, args):
        fn = Update(original, new).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, ret, chm) = fn(key, *in_args)
        old = original.get_choices()
        discard = old.setdiff(chm)
        return key, (
            w,
            Trace(args, tuple(ret), chm, original.get_score() + w),
            discard,
        )

    return lambda key, chm, new, args: _inner(key, chm, new, args)


def arg_grad(f, argnums):
    def _inner(key, tr, argnums, args):
        fn = ArgumentGradients(tr, argnums).transform(f)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        arg_grads, key = fn(key, *in_args)
        return key, arg_grads

    return lambda key, tr, args: _inner(key, tr, argnums, args)


def choice_grad(f):
    def _inner(key, tr, chm, args):
        fn = ChoiceGradients(tr).transform(f)(key, *args)
        choice_grads, key = fn(key, chm)
        return key, choice_grads

    return lambda key, tr, chm, args: _inner(key, tr, chm, args)
