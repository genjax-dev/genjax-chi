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
Module for a primitive handling `Jaxpr` interpreter.

Supports zero-cost effect handling implementations of each of the
generative function interfaces -- dispatching on `gen_fn_p`
primitives for generative function calls.
"""

import abc
import dataclasses
from typing import Any
from typing import Dict

import jax
import jax.core as jc
import jax.tree_util as jtu
from jax.util import safe_map

from genjax.core.hashabledict import hashabledict
from genjax.core.specialization import concrete_cond
from genjax.generative_functions.builtin.builtin_datatypes import (
    BuiltinChoiceMap,
)
from genjax.generative_functions.builtin.intrinsics import gen_fn_p


@dataclasses.dataclass
class Handler:
    handles: jc.Primitive

    def _transform(self, f, *args, **kwargs):
        expr = jax.make_jaxpr(f, **kwargs)(*args)
        fn = handle(self, expr)
        return fn

    def transform(self, f, **kwargs):
        return lambda *args: self._transform(f, *args, **kwargs)

    @abc.abstractmethod
    def handle(self, *args, **kwargs):
        pass


#####
# Primitive-handling interpreter
#####


def eval_jaxpr_handler(
    handler: Handler,
    jaxpr: jc.Jaxpr,
    consts,
    *args,
):
    env: Dict[jc.Var, Any] = {}

    def write(v, val):
        env[v] = val

    safe_map(write, jaxpr.constvars, consts)

    # This is the recursion that replaces the main loop in the original
    # `eval_jaxpr`.
    def eval_jaxpr_recurse(eqns, env, invars, args):
        # The handler could call the continuation multiple times so we
        # we need this function to be somewhat pure. We copy `env` to
        # ensure it isn't mutated.
        env = env.copy()

        def read(v):
            if isinstance(v, jc.Literal):
                return v.val
            else:
                return env[v]

        def write(v, val):
            env[v] = val

        safe_map(write, invars, args)

        if eqns:
            eqn = eqns[0]
            kwargs = eqn.params
            in_vals = safe_map(read, eqn.invars)
            in_vals = list(in_vals)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            if hasattr(eqn.primitive, "must_handle"):
                args = subfuns + in_vals
                # This definition "reifies" the remainder of the evaluation
                # loop so it can be explicitly passed to the handler.

                def continuation(*args):
                    return eval_jaxpr_recurse(
                        eqns[1:], env, eqn.outvars, [*args]
                    )

                assert eqn.primitive == handler.handles
                return handler.handle(continuation, *args, **kwargs)

            ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
            if not eqn.primitive.multiple_results:
                ans = [ans]

            return eval_jaxpr_recurse(eqns[1:], env, eqn.outvars, ans)
        else:
            return safe_map(read, jaxpr.outvars)

    return eval_jaxpr_recurse(jaxpr.eqns, env, jaxpr.invars, args)


# Our special interpreter -- allows us to dispatch with primitives,
# and implements directed CPS-style code generation strategy.
def I_prime(handler, f):
    return lambda *xs: eval_jaxpr_handler(
        handler,
        f.jaxpr,
        f.literals,
        *xs,
    )


def handle(handler, expr):
    """
    Sugar: Abstract interpret a :code:`Jaxpr` with a
    :code:`handler: Sequence[Handler]`
    """
    return I_prime(handler, expr)


######################################
#  Generative function interpreters  #
######################################


class Simulate(Handler):
    def __init__(self):
        self.handles = gen_fn_p
        self.state = BuiltinChoiceMap(hashabledict())
        self.score = 0.0
        self.return_or_continue = False

    def handle(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)
        key, tr = gen_fn.simulate(key, args, **kwargs)
        score = tr.get_score()
        v = tr.get_retval()
        self.state[addr] = tr
        self.score += score

        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            key, ret = f(key, v)
            return key, (ret, self.state, self.score)


def handler_simulate(f, **kwargs):
    def _inner(key, args):
        fn = Simulate().transform(f, **kwargs)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (retval, chm, score) = fn(key, *in_args)
        return key, (f, args, retval, chm, score)

    return _inner


class Importance(Handler):
    def __init__(self, constraints):
        self.handles = gen_fn_p
        self.state = BuiltinChoiceMap(hashabledict())
        self.score = 0.0
        self.weight = 0.0
        self.constraints = constraints
        self.return_or_continue = False

    def handle(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
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
            return f(key, v)
        else:
            self.return_or_continue = True
            key, ret = f(key, v)
            return key, (self.weight, ret, self.state, self.score)


def handler_importance(f, **kwargs):
    def _inner(key, chm, args):
        fn = Importance(chm).transform(f, **kwargs)(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        key, (w, retval, chm, score) = fn(key, *in_args)
        return key, (w, (f, args, retval, chm, score))

    return _inner


############################################
#  Automatic differentiation interpreters  #
############################################

# This section is particularly complex.
# TODO: Extensive comments.

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
    key, choice_vjp = gen_fn.choice_vjp(
        key,
        trace,
        selection,
    )
    trace_grads, arg_grads = choice_vjp(v_retval_grad)
    return (None, trace_grads, None, arg_grads)


choice_pretend_generative_function_call.defvjp(
    choice_pretend_fwd, choice_pretend_bwd
)


class ChoiceGradients(Handler):
    def __init__(self, source, selection):
        self.handles = gen_fn_p
        self.source = source
        self.selection = selection
        self.score = 0.0
        self.return_or_continue = False

    def handle(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
        args = jtu.tree_unflatten(args_form, args)
        sub_trace = self.source.get_subtree(addr)
        if self.selection.has_subtree(addr):
            sub_selection = self.selection.get_subtree(addr)
            key, (w, v) = choice_pretend_generative_function_call(
                key, sub_trace, sub_selection, args
            )
        else:
            key, (w, sub_trace) = gen_fn.importance(
                key, sub_trace.get_choices(), args
            )
            v = sub_trace.get_retval()

        self.score += w
        if self.return_or_continue:
            return f(key, *v)
        else:
            self.return_or_continue = True
            key, *ret = f(key, *v)
            return key, ret, self.score


def handler_choice_grad(f, key, selection, **kwargs):
    def _inner(tr, args):
        handler = ChoiceGradients(tr, selection).transform(f, **kwargs)
        fn = handler(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        new_key, v, score = fn(key, *in_args)
        return (score, tuple(v)), new_key

    return _inner


# Adjoint root, for retval gradients.
# NOTE: the order of decorators here matters.
# Define custom JVPs first, then custom VJPs.
@jax.custom_vjp
@jax.custom_jvp
def retval_pretend_generative_function_call(key, trace, selection, args):
    return key, trace.get_retval()


# Custom JVPs.
def retval_pretend_jvp_fwd(primals, tangents):
    pass


retval_pretend_generative_function_call.defjvp(retval_pretend_jvp_fwd)

# Custom VJPs.
def retval_pretend_vjp_fwd(key, trace, selection, args):
    ret = retval_pretend_generative_function_call(key, trace, selection, args)
    key, v = ret
    key, sub_key = jax.random.split(key)
    return (key, v), (sub_key, trace, selection)


def retval_pretend_vjp_bwd(res, retval_grad):
    key, trace, selection = res
    gen_fn = trace.get_gen_fn()
    _, v_retval_grad = retval_grad
    key, retval_vjp = gen_fn.retval_vjp(
        key,
        trace,
        selection,
    )
    trace_grads, arg_grads = retval_vjp(v_retval_grad)
    return (None, trace_grads, None, arg_grads)


retval_pretend_generative_function_call.defvjp(
    retval_pretend_vjp_fwd, retval_pretend_vjp_bwd
)


class RetvalGradients(Handler):
    def __init__(self, source, selection):
        self.handles = gen_fn_p
        self.source = source
        self.selection = selection
        self.return_or_continue = False

    def handle(self, f, key, *args, addr, gen_fn, args_form, **kwargs):
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


def handler_retval_grad(f, key, selection, **kwargs):
    def _inner(tr, args):
        handler = RetvalGradients(tr, selection).transform(f, **kwargs)
        fn = handler(key, *args)
        in_args, _ = jtu.tree_flatten(args)
        new_key, ret = fn(key, *in_args)
        return tuple(ret), new_key

    return _inner
