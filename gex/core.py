# This is a sample implementation of Gen concepts
# using zero-cost effect staging on top of JAX.

import jax
import jax.numpy as jnp
from jax import make_jaxpr, core
from jax.util import safe_map, safe_zip
from jax._src import abstract_arrays
import jax.random as random
import jax.scipy.stats as stats
from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
)

map = safe_map
zip = safe_zip

#####
##### Handlers
#####


# A handler dispatchs a `jax.core.Primitive` - and must provide
# a `Callable` with signature `def (name_of_primitive)(continuation, *args)`
# where `*args` must match the `core.Primitive` declaration
# signature.
class Handler:
    handles: Sequence[core.Primitive]
    callable: Callable

    def __init__(self, handles: Sequence[core.Primitive], callable: Callable):
        self.handles = handles
        self.callable = callable


#####
##### Interpreter with handlers
#####


# This is an interpreter which is parametrized by a handler stack.
# The handler stack is consulted when a `core.Primitive` with a `must_handle`
# attribute is encountered.
#
# This interpreter should always be staged out -- so it should be handling primitives is a zero runtime cost process.
def eval_jaxpr_handler(
    handler_stack: Sequence[Handler], jaxpr: core.Jaxpr, consts, *args
):
    env: Dict[Var, Any] = {}

    def write(v, val):
        env[v] = val

    map(write, jaxpr.constvars, consts)

    # This is the recursion that replaces the main loop in the original
    # `eval_jaxpr`.
    def eval_jaxpr_loop(eqns, env, invars, args):
        # The handler could call the continuation multiple times so we
        # we need this function to be somewhat pure. We copy `env` to
        # ensure it isn't mutated.
        env = env.copy()

        def read(v):
            if type(v) is core.Literal:
                return v.val
            else:
                return env[v]

        def write(v, val):
            env[v] = val

        map(write, invars, args)

        if eqns:
            eqn = eqns[0]

            # Here's where we encode `prim` and `addr` for `trace`.
            kwargs = eqn.params

            in_vals = map(read, eqn.invars)
            in_vals = list(in_vals)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            if hasattr(eqn.primitive, "must_handle"):
                args = subfuns + in_vals
                # This definition "reifies" the remainder of the evaluation
                # loop so it can be explicitly passed to the handler.
                def continuation(*args):
                    return eval_jaxpr_loop(eqns[1:], env, eqn.outvars, [*args])

                for handler in reversed(handler_stack):
                    if eqn.primitive in handler.handles:
                        # The handler must provide a method with
                        # the name of the primitive.
                        callable = getattr(handler, repr(eqn.primitive))
                        return callable(continuation, *args, **kwargs)

            ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
            if not eqn.primitive.multiple_results:
                ans = [ans]
            return eval_jaxpr_loop(eqns[1:], env, eqn.outvars, ans)
        else:
            return map(read, jaxpr.outvars)

    return eval_jaxpr_loop(jaxpr.eqns, env, jaxpr.invars, args)


#####
##### Transformers
#####


# Takes `Callable(*args)` to `Jaxpr`.
def T(*xs):
    return lambda f: make_jaxpr(f)(*xs)


# The usual interpreter
def I(f):
    # `make_jaxpr` builds a separate "symbol table" containing the constants
    # needed by the jaxpr. This is why we also pass `f.literals` into
    # `eval_jaxpr`.
    return lambda *xs: jax.core.eval_jaxpr(f.jaxpr, f.literals, *xs)


# Our special interpreter
def I_prime(handler_stack, f):
    return lambda *xs: eval_jaxpr_handler(handler_stack, f.jaxpr, f.literals, *xs)


# Sugar: lift a `Callable(*args)` to `Jaxpr`
def lift(f, *args):
    return T(*args)(f)


# Sugar: Abstract interpret a `Jaxpr` with a `handler_stack :: List Handler`
def handle(handler_stack, expr):
    return I_prime(handler_stack, expr)


# Sugar: JIT a `Jaxpr` with `*args`
def jit(expr, *args):
    return jax.jit(expr)(*args)
