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

import jax
from .core import Handler, handle, lift
from .intrinsics import trace_p, splice_p, unsplice_p

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
        self.handles = [trace_p, splice_p, unsplice_p]

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        prim = prim()
        key, v = prim.sample(*args, **kwargs)
        return f(key, v)

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        return f(())

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        return f(())

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # JIT compile a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        jitted = jax.jit(expr)
        return jitted

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


class Simulate(Handler):
    def __init__(self):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.state = {}
        self.scores = {}
        self.level = []
        self.score = 0.0
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        prim = prim()
        key, v = prim.sample(*args, **kwargs)
        self.state[(*self.level, addr)] = v
        score = prim.score(v, *args[1:])
        self.scores[(*self.level, addr)] = score
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            ret = f(key, v)
            return (ret, self.state, self.scores, self.score)

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

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # JIT compile a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        jitted = jax.jit(expr)
        return jitted

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


class Importance(Handler):
    def __init__(self, choice_map):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.state = {}
        self.level = []
        self.scores = {}
        self.score = 0.0
        self.weight = 0.0
        self.obs = choice_map
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        if (*self.level, addr) in self.obs:
            v = self.obs.get((*self.level, addr))
            score = prim.score(v, *args[1:])
            self.scores[(*self.level, addr)] = score
            self.state[(*self.level, addr)] = v
            self.weight += score
        else:
            key, v = prim.sample(*args, **kwargs)
            self.state[(*self.level, addr)] = v
            score = prim.score(v, *args[1:])
            self.scores[(*self.level, addr)] = score
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            ret = f(key, v)
            return (self.weight, ret, self.state, self.scores, self.score)

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

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # JIT compile a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        jitted = jax.jit(expr)
        return jitted

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


class Diff(Handler):
    def __init__(self, original, new):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.level = []
        self.weight = 0.0
        self.original = original
        self.choice_change = new
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        v = self.original.get_choices().get((*self.level, addr))
        if (*self.level, addr) in self.choice_change:
            v = self.choice_change.get((*self.level, addr))
            forward = prim.score(v, *args[1:])
            backward = self.original.scores[(*self.level, addr)]
            self.weight += forward - backward
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            ret = f(key, v)
            return self.weight, ret

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

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # JIT compile a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        jitted = jax.jit(expr)
        return jitted

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


class Update(Handler):
    def __init__(self, original, new):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.level = []
        self.state = {}
        self.scores = {}
        self.weight = 0.0
        self.original = original
        self.choice_change = new
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        if (*self.level, addr) in self.choice_change:
            v = self.choice_change.get((*self.level, addr))
            forward = prim.score(v, *args[1:])
            backward = self.original.scores[(*self.level, addr)]
            self.scores[(*self.level, addr)] = forward
            self.state[(*self.level, addr)] = v
            self.weight += forward - backward
        else:
            v = self.original.get_choices().get((*self.level, addr))
            self.state[(*self.level, addr)] = v
            self.scores[(*self.level, addr)] = self.original.scores[
                (*self.level, addr)
            ]
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            ret = f(key, v)
            return self.weight, ret, self.scores, self.state

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

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # JIT compile a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        jitted = jax.jit(expr)
        return jitted

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


class ArgumentGradients(Handler):
    def __init__(self, tr, argnums):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.argnums = argnums
        self.level = []
        self.score = 0.0
        self.tr = tr
        self.sources = tr.get_choices()
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, addr, prim, **kwargs):
        key = args[0]
        prim = prim()
        v = self.sources.get((*self.level, addr))
        score = prim.score(v, *args[1:])
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            _ = f(key, v)
            return self.score

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

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    def stage(self, f):
        return lambda *args: self._stage(f, *args)

    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = jax.grad(expr, self.argnums)
        jitted = jax.jit(expr)
        return jitted

    def jit(self, f):
        return lambda *args: self._jit(f, *args)


class ChoiceGradients(Handler):
    def __init__(self, tr):
        self.tr = tr

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):

        # Here, we define a utility function `diff` which
        # interprets with the `Diff` handler.
        def diff(chm):
            handler = Diff(self.tr, chm)
            expr = lift(f, *args)
            fn = handle([handler], expr)
            w, _ = fn(*args)
            return self.tr.get_score() + w

        # Here, because of composition -- we can lift `diff` itself,
        # and now we have a function parametrized by `chm`
        # which we can compute the gradient of.
        #
        # This function computes the logpdf (score) of the trace,
        # but it accepts the seed `chm` as an argument -- and then stages out
        # the gradient computation using `jax.grad`.
        return lambda chm: lift(jax.grad(diff), chm)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)

    def _jit(self, f, *args):
        def diff(chm):
            handler = Diff(self.tr, chm)
            expr = lift(f, *args)
            fn = handle([handler], expr)
            w, _ = fn(*args)
            return self.tr.get_score() + w

        jitted = jax.jit(jax.grad(diff))
        return jitted

    def jit(self, f):
        return lambda *args: self._jit(f, *args)
