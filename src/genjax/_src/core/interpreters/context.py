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
"""This module contains a transformation infrastructure based on interpreters
with stateful contexts and custom primitive handling lookups."""

import abc
import copy
import dataclasses
import functools
from contextlib import contextmanager

import jax.core as jc
import jax.tree_util as jtu
from jax import api_util
from jax import util as jax_util
from jax.extend import linear_util as lu

from genjax._src.core.interpreters import staging
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Iterable
from genjax._src.core.typing import List
from genjax._src.core.typing import Type
from genjax._src.core.typing import Union
from genjax._src.core.typing import Value


################################
# Traces, tracers, and context #
################################


class ContextualTracer(jc.Tracer):
    """A `ContextualTracer` encapsulates a single value."""

    def __init__(self, trace: "ContextualTrace", val: Value):
        self._trace = trace
        self.val = val

    @property
    def aval(self):
        return jc.raise_to_shaped(jc.get_aval(self.val))

    def full_lower(self):
        return self


class ContextualTrace(jc.Trace):
    """An evaluating trace that dispatches to a dynamic context."""

    def pure(self, val: Value) -> ContextualTracer:
        return ContextualTracer(self, val)

    def sublift(self, tracer: ContextualTracer) -> ContextualTracer:
        return self.pure(tracer.val)

    def lift(self, val: Value) -> ContextualTracer:
        return self.pure(val)

    def process_primitive(
        self,
        primitive: jc.Primitive,
        tracers: List[ContextualTracer],
        params: Dict[str, Any],
    ) -> Union[ContextualTracer, List[ContextualTracer]]:
        context = staging.get_dynamic_context(self)
        custom_rule = context.get_custom_rule(primitive)
        if custom_rule:
            return custom_rule(self, *tracers, **params)
        return self.default_process_primitive(primitive, tracers, params)

    def default_process_primitive(
        self,
        primitive: jc.Primitive,
        tracers: List[ContextualTracer],
        params: Dict[str, Any],
    ) -> Union[ContextualTracer, List[ContextualTracer]]:
        context = staging.get_dynamic_context(self)
        vals = [v.val for v in tracers]
        if context.can_process(primitive):
            outvals = context.process_primitive(primitive, *vals, **params)
            return jax_util.safe_map(self.pure, outvals)
        subfuns, params = primitive.get_bind_params(params)
        args = subfuns + vals
        outvals = primitive.bind(*args, **params)
        if not primitive.multiple_results:
            outvals = [outvals]
        out_tracers = jax_util.safe_map(self.full_raise, outvals)
        if primitive.multiple_results:
            return out_tracers
        return out_tracers[0]

    def process_call(
        self,
        call_primitive: jc.Primitive,
        f: Any,
        tracers: List[ContextualTracer],
        params: Dict[str, Any],
    ):
        context = staging.get_dynamic_context(self)
        return context.process_higher_order_primitive(
            self, call_primitive, f, tracers, params, False
        )

    def post_process_call(self, call_primitive, out_tracers, params):
        vals = tuple(t.val for t in out_tracers)
        master = self.main

        def todo(x):
            trace = ContextualTrace(master, jc.cur_sublevel())
            return jax_util.safe_map(functools.partial(ContextualTracer, trace), x)

        return vals, todo

    def process_map(
        self,
        call_primitive: jc.Primitive,
        f: Any,
        tracers: List[ContextualTracer],
        params: Dict[str, Any],
    ):
        context = staging.get_dynamic_context(self)
        return context.process_higher_order_primitive(
            self, call_primitive, f, tracers, params, True
        )

    post_process_map = post_process_call

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
        context = staging.get_dynamic_context(self)
        return context.process_custom_jvp_call(
            self, primitive, fun, jvp, tracers, symbolic_zeros=symbolic_zeros
        )

    def post_process_custom_jvp_call(self, out_tracers, jvp_was_run):
        context = staging.get_dynamic_context(self)
        return context.post_process_custom_jvp_call(self, out_tracers, jvp_was_run)

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        context = staging.get_dynamic_context(self)
        return context.process_custom_vjp_call(
            self, primitive, fun, fwd, bwd, tracers, out_trees
        )

    def post_process_custom_vjp_call(self, out_tracers, params):
        context = staging.get_dynamic_context(self)
        return context.post_process_custom_vjp_call(self, out_tracers, params)

    def post_process_custom_vjp_call_fwd(self, out_tracers, out_trees):
        context = staging.get_dynamic_context(self)
        return context.post_process_custom_vjp_call_fwd(self, out_tracers, out_trees)


@dataclasses.dataclass
class Context(Pytree):
    @abc.abstractmethod
    def yield_state(self):
        pass

    @abc.abstractmethod
    def get_custom_rule(self, primitive):
        raise NotImplementedError

    @abc.abstractmethod
    def can_process(self, primitive):
        raise NotImplementedError

    @abc.abstractmethod
    def process_primitive(
        self,
        primitive,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def process_higher_order_primitive(
        self,
        trace: ContextualTrace,
        call_primitive: jc.Primitive,
        f: Any,
        tracers: List[ContextualTracer],
        params: Dict[str, Any],
        is_map: bool,
    ):
        raise NotImplementedError

    # TODO: got this impl from the partial evaluation interpreter,
    # is it correct?
    def process_custom_jvp_call(
        self, trace, primitive, fun, jvp, tracers, *, symbolic_zeros
    ):
        return fun.call_wrapped(*tracers)

    def post_process_custom_jvp_call(self, trace, out_tracers, jvp_was_run):
        raise NotImplementedError

    def process_custom_vjp_call(
        self, trace, primitive, fun, fwd, bwd, tracers, out_trees
    ):
        raise NotImplementedError

    def post_process_custom_vjp_call(self, trace, out_tracers, params):
        raise NotImplementedError

    def post_process_custom_vjp_call_fwd(self, trace, out_tracers, out_trees):
        raise NotImplementedError


################
# Interpreters #
################

VarOrLiteral = Union[jc.Var, jc.Literal]


class Environment:
    """Keeps track of variables and their values during propagation."""

    def __init__(self):
        self.env: Dict[jc.Var, Value] = {}

    def read(self, var: VarOrLiteral) -> Value:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            return self.env.get(var)

    def write(self, var: VarOrLiteral, cell: Value) -> Value:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.read(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var] = cell
        return self.env[var]

    def __getitem__(self, var: VarOrLiteral) -> Value:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jc.Literal):
            return True
        return var in self.env

    def copy(self):
        return copy.copy(self)


@dataclasses.dataclass
class ForwardInterpreter(Pytree):
    def flatten(self):
        return (), ()

    # This produces an instance of `Interpreter`
    # as a context manager - to allow us to control stack traces,
    # if required.
    @classmethod
    @contextmanager
    def new(cls):
        try:
            yield ForwardInterpreter()
        except Exception as e:
            raise e

    def _eval_jaxpr(
        self,
        trace_type: Type[jc.Trace],
        main: jc.MainTrace,
        jaxpr: jc.Jaxpr,
        consts: List[Value],
        args: List[Value],
    ):
        env = Environment()
        trace = trace_type(main, jc.cur_sublevel())
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, args)
        for eqn in jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            outvals = eqn.primitive.bind_with_trace(trace, invals, eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)
        return jax_util.safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, trace_type, main, fn, *args, **kwargs):
        closed_jaxpr, (flat_args, _, out_tree) = staging.stage(fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr(trace_type, main, jaxpr, consts, flat_args)
        out = jtu.tree_unflatten(out_tree, flat_out)
        return out


Fwd = ForwardInterpreter


@dataclasses.dataclass
class CPSInterpreter(Pytree):
    def flatten(self):
        return (), ()

    # This produces an instance of `Interpreter`
    # as a context manager - to allow us to control stack traces,
    # if required.
    @classmethod
    @contextmanager
    def new(cls):
        try:
            yield CPSInterpreter()
        except Exception as e:
            raise e

    def _eval_jaxpr_cps(
        self,
        trace_type: Type[jc.Trace],
        main: jc.MainTrace,
        jaxpr: jc.Jaxpr,
        consts: List[Value],
        args: List[Value],
    ):
        env = Environment()
        trace = trace_type(main, jc.cur_sublevel())
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, args)

        def eval_jaxpr_recurse(eqns, env, invars, args):
            # The rule could call the continuation multiple times so we
            # we need this function to be somewhat pure.
            # We copy `env` to ensure it isn't mutated.
            env = env.copy()
            jax_util.safe_map(env.write, invars, args)

            if eqns:
                eqn = eqns[0]
                in_vals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + in_vals

                # Create a continuation to pass to bind.
                def kont(*args):
                    return eval_jaxpr_recurse(eqns[1:], env, eqn.outvars, [*args])

                # Pass all the information over to the handler,
                # which gets to choose how to interpret the primitive.
                outvals = eqn.primitive.bind_with_trace(
                    trace, args, {"kont": kont, **params}
                )
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]

                return jtu.tree_leaves(outvals)

            return jax_util.safe_map(env.read, jaxpr.outvars)

        return eval_jaxpr_recurse(jaxpr.eqns, env, jaxpr.invars, args)

    def run_interpreter(self, trace_type, main, kont, fn, *args, **kwargs):
        def _inner(*args, **kwargs):
            return kont(fn(*args, **kwargs))

        closed_jaxpr, (flat_args, _, out_tree) = staging.stage(_inner)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_cps(trace_type, main, jaxpr, consts, flat_args)
        out = jtu.tree_unflatten(out_tree, flat_out)
        return out


Cont = CPSInterpreter

##########################
# Linear transformations #
##########################

# This is a convenient interface to use the interpreter.
#
# Arguments get lifted to ContextualTracers and passed in
# to the interpreter evaluation.
#
# The trace, tracers, and context explicitly control the behavior
# of primitive bind invocations inside the interpreter loop.

###############
# No tangents #
###############


# TODO: be explicit about the `get_value` interface.
def _unwrap_tracer(v):
    if isinstance(v, ContextualTracer):
        return v.val
    else:
        return v


@lu.transformation
def _transform(
    trace_type: Type[jc.Trace],
    main: jc.MainTrace,
    ctx: Context,
    args: Iterable[Any],
):
    trace = trace_type(main, jc.cur_sublevel())
    in_tracers = jax_util.safe_map(trace.full_raise, args)
    with staging.new_dynamic_context(main, ctx):
        ans = yield in_tracers, {}
        out_tracers = jax_util.safe_map(trace.full_raise, ans)
        stateful_tracers = ctx.yield_state()
        del main
    (
        out_values,
        stateful_values,
    ) = jtu.tree_map(_unwrap_tracer, (out_tracers, stateful_tracers))
    yield out_values, stateful_values


def transform(f, ctx: Context, trace_type: Type[jc.Trace] = ContextualTrace):
    # Runs the interpreter.
    def _run_interpreter(main, *args, **kwargs):
        with Fwd.new() as interpreter:
            return interpreter.run_interpreter(trace_type, main, f, *args, **kwargs)

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        with jc.new_main(trace_type) as main:
            fun = lu.wrap_init(functools.partial(_run_interpreter, main), kwargs)
            flat_args, in_tree = jtu.tree_flatten(args)
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
            flat_fun = _transform(flat_fun, trace_type, main, ctx)
            out_flat, ctx_statefuls = flat_fun.call_wrapped(flat_args)
            del main
        return jtu.tree_unflatten(out_tree(), out_flat), ctx_statefuls

    return wrapped


#####
# CPS transform
#####


def cps_transform(f, ctx: Context, trace_type: Type[jc.Trace] = ContextualTrace):
    # Runs the interpreter.
    def _run_interpreter(main, kont, *args, **kwargs):
        with Cont.new() as interpreter:
            return interpreter.run_interpreter(
                trace_type, main, kont, f, *args, **kwargs
            )

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(kont, *args, **kwargs):
        with jc.new_main(trace_type) as main:
            fun = lu.wrap_init(functools.partial(_run_interpreter, main, kont), kwargs)
            flat_args, in_tree = jtu.tree_flatten(args)
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
            flat_fun = _transform(flat_fun, trace_type, main, ctx)
            out_flat, ctx_statefuls = flat_fun.call_wrapped(flat_args)
            del main
        return jtu.tree_unflatten(out_tree(), out_flat), ctx_statefuls

    return wrapped
