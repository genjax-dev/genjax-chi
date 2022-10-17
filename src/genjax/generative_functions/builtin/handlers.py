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

from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.hashabledict import hashabledict
from genjax.core.masks import BooleanMask
from genjax.core.specialization import concrete_and
from genjax.core.specialization import concrete_cond
from genjax.generative_functions.builtin.builtin_datatypes import (
    BuiltinChoiceMap,
)
from genjax.generative_functions.builtin.handling import Handler
from genjax.generative_functions.builtin.intrinsics import gen_fn_p


#####
# GFI handlers
#####

# Note:
#
# These handlers _do not manipulate runtime values_ --
# the pointers they hold to objects like `v` and `score` are JAX `Tracer`
# values. When we do computations with these values,
# it adds to the `Jaxpr` trace.
#
# So the trick is write `trace` in each handler
# to coerce the resulting `Jaxpr` to send out the accumulated state we want.

# One other note:
#
# In the code of each handler, you might see `tree_unflatten` and
# `args_form` -- this is to coerce flattened `Pytree` representations
# across JAX jit boundaries -- and allow usage of arbitrary `Pytree`
# types as arguments to GFI calls.

# You'll find that this implementation mimics many aspects of
# Gen's dynamic DSL -- but it's staged out (by the above Jaxpr discussion)
# the result is that there's no lookup, storage, or call overhead,
# it's all pure inlined array code.


class Simulate(Handler):
    def __init__(self):
        self.handles = [
            gen_fn_p,
        ]
        self.state = BuiltinChoiceMap(hashabledict())
        self.score = 0.0
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)
        key, tr = gen_fn.simulate(key, args, **kwargs)
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


class Importance(Handler):
    def __init__(self, constraints):
        self.handles = [
            gen_fn_p,
        ]
        self.state = BuiltinChoiceMap(hashabledict())
        self.score = 0.0
        self.weight = 0.0
        self.constraints = constraints
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)

        def _simulate_branch(key, args):
            key, tr = gen_fn.simulate(key, args, **kwargs)
            return key, (0.0, tr)

        def _importance_branch(key, args):
            submap = self.constraints.get_subtree(addr)
            key, (w, tr) = gen_fn.importance(key, submap, args, **kwargs)
            return key, (w, tr)

        check = self.constraints.has_subtree(addr)
        key, (w, tr) = concrete_cond(
            check,
            _importance_branch,
            _simulate_branch,
            key,
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


class Update(Handler):
    def __init__(self, prev, new):
        self.handles = [
            gen_fn_p,
        ]
        self.state = BuiltinChoiceMap(hashabledict())
        self.discard = BuiltinChoiceMap(hashabledict())
        self.weight = 0.0
        self.prev = prev
        self.choice_change = new
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)
        has_previous = self.prev.has_subtree(addr)
        constrained = self.choice_change.has_subtree(addr)

        def _update_branch(key, args):
            prev_tr = self.prev.get_subtree(addr)
            chm = self.choice_change.get_subtree(addr)
            key, (w, tr, discard) = gen_fn.update(
                key, prev_tr, chm, args, **kwargs
            )
            discard = discard.strip_metadata()
            return key, (w, tr, discard)

        def _has_prev_branch(key, args):
            prev_tr = self.prev.get_subtree(addr)
            key, (w, _, _) = gen_fn.update(
                key, prev_tr, EmptyChoiceMap(), args, **kwargs
            )
            discard = BooleanMask.new(False, prev_tr.strip_metadata())
            return key, (w, prev_tr, discard)

        def _constrained_branch(key, args):
            prev_tr = self.prev.get_subtree(addr)
            chm = self.choice_change.get_subtree(addr)
            key, (w, tr) = gen_fn.importance(key, chm, args, **kwargs)
            discard = BooleanMask.new(False, prev_tr.strip_metadata())
            return key, (w, tr, discard)

        key, (w, tr, discard) = concrete_cond(
            concrete_and(has_previous, constrained),
            _update_branch,
            lambda key, args: concrete_cond(
                has_previous,
                _has_prev_branch,
                _constrained_branch,
                key,
                args,
            ),
            key,
            args,
        )

        self.weight += w
        self.state[addr] = tr
        self.discard[addr] = discard
        v = tr.get_retval()

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, (self.weight, ret, self.state, self.discard)


class ArgumentGradients(Handler):
    def __init__(self, tr, argnums):
        self.handles = [
            gen_fn_p,
        ]
        self.argnums = argnums
        self.score = 0.0
        self.source = tr.get_choices()
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)
        has_source = self.source.has_subtree(addr)

        def _has_source_branch(key, args):
            sub_tr = self.source.get_subtree(addr)
            chm = sub_tr.get_choices()
            key, (w, tr) = gen_fn.importance(key, chm, args, **kwargs)
            v = tr.get_retval()
            return (w, v)

        def _no_source_branch(key, args):
            key, tr = gen_fn.simulate(key, args, **kwargs)
            v = tr.get_retval()
            return (0.0, v)

        w, v = concrete_cond(
            has_source,
            _has_source_branch,
            _no_source_branch,
            key,
            args,
        )
        self.score += w

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *_ = f(key, *v)
            return self.score, key


# Adjoint root, for choice gradients.
@jax.custom_vjp
def choice_pretend_generative_function_call(key, trace, selection, args):
    return key, (trace.get_score(), trace.get_retval())


def choice_pretend_fwd(key, trace, selection, args):
    ret = choice_pretend_generative_function_call(key, trace, selection, args)
    key, (w, v) = ret
    key, sub_key = jax.random.split(key)
    return (key, (w, v)), (sub_key, trace, selection)


def choice_pretend_bwd(res, retval_grad):
    key, trace, selection = res
    gen_fn = trace.get_gen_fn()
    _, (_, v_retval_grad) = retval_grad
    _, (trace_grads, arg_grads) = gen_fn.choice_grad(
        key, trace, selection, v_retval_grad
    )
    return (None, trace_grads, None, arg_grads)


choice_pretend_generative_function_call.defvjp(
    choice_pretend_fwd, choice_pretend_bwd
)


class ChoiceGradients(Handler):
    def __init__(self, source, selection):
        self.handles = [
            gen_fn_p,
        ]
        self.source = source
        self.selection = selection
        self.score = 0.0
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)
        sub_selection = self.selection.get_subtree(addr)
        sub_trace = self.source.get_subtree(addr)
        key, (w, v) = choice_pretend_generative_function_call(
            key, sub_trace, sub_selection, args
        )
        self.score += w

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, ret, self.score


# Adjoint root, for retval gradients.
@jax.custom_vjp
def retval_pretend_generative_function_call(key, trace, selection, args):
    return key, trace.get_retval()


def retval_pretend_fwd(key, trace, selection, args):
    ret = retval_pretend_generative_function_call(key, trace, selection, args)
    key, v = ret
    key, sub_key = jax.random.split(key)
    return (key, v), (sub_key, trace, selection)


def retval_pretend_bwd(res, retval_grad):
    key, trace, selection = res
    gen_fn = trace.get_gen_fn()
    _, v_retval_grad = retval_grad
    _, (trace_grads, arg_grads) = gen_fn.retval_grad(
        key, trace, selection, v_retval_grad
    )
    return (None, trace_grads, None, arg_grads)


retval_pretend_generative_function_call.defvjp(
    retval_pretend_fwd, retval_pretend_bwd
)


class RetvalGradients(Handler):
    def __init__(self, source, selection):
        self.handles = [
            gen_fn_p,
        ]
        self.source = source
        self.selection = selection
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)
        sub_trace = self.source.get_subtree(addr)
        if self.selection.has_subtree(addr):
            sub_selection = self.selection.get_subtree(addr)
            key, v = retval_pretend_generative_function_call(
                key, sub_trace, sub_selection, args
            )
        else:
            v = sub_trace.get_retval()

        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, ret


#####
# Generative function interface
#####


def handler_simulate(f, **kwargs):
    def _inner(key, args):
        fn = Simulate().transform(f, **kwargs)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (r, chm, score) = fn(key, *in_args)
        return key, (f, args, tuple(r), chm, score)

    return _inner


def handler_importance(f, **kwargs):
    def _inner(key, chm, args):
        fn = Importance(chm).transform(f, **kwargs)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, r, chm, score) = fn(key, *in_args)
        return key, (w, (f, args, tuple(r), chm, score))

    return _inner


def handler_update(f, **kwargs):
    def _inner(key, prev, new, args):
        fn = Update(prev, new).transform(f, **kwargs)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, ret, chm, discard) = fn(key, *in_args)
        return key, (
            w,
            (f, args, tuple(ret), chm, prev.get_score() + w),
            discard,
        )

    return _inner


def handler_arg_grad(f, argnums, **kwargs):
    def _inner(key, tr, args):
        fn = ArgumentGradients(tr, argnums).transform(f, **kwargs)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        arg_grads, key = fn(key, *in_args)
        return key, arg_grads

    return _inner


def handler_choice_grad(f, key, selection, **kwargs):
    def _inner(tr, args):
        handler = ChoiceGradients(tr, selection).transform(f, **kwargs)
        fn = handler(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        new_key, v, score = fn(key, *in_args)
        return (score, tuple(v)), new_key

    return _inner


def handler_retval_grad(f, key, selection, **kwargs):
    def _inner(tr, args):
        handler = RetvalGradients(tr, selection).transform(f, **kwargs)
        fn = handler(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        new_key, ret = fn(key, *in_args)
        return tuple(ret), new_key

    return _inner
