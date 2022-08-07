import jax
import jax.numpy as jnp
from jax import grad, jit, make_jaxpr, vmap, core
from jax.util import safe_map, safe_zip
from jax._src import abstract_arrays
import jax.random as random
from jax import linear_util as lu
from jax._src import source_info_util
import functools
from functools import partial, partialmethod, total_ordering
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    Iterable,
    Hashable,
)

map = safe_map
zip = safe_zip

#####
##### Handlers
#####


class Handler:
    handles: core.Primitive
    callable: Callable

    def __init__(self, handles: core.Primitive, callable: Callable):
        self.handles = handles
        self.callable = callable


#####
##### Interpreter with handlers
#####


# This is an interpreter which is parametrized by a handler stack.
# The handler stack is consorted when a `core.Primitive` with a `must_handle`
# attribute is encountered.
#
# This interpreter should always be staged out -- so it should be zero-cost.
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
            in_vals = map(read, eqn.invars)
            in_vals = list(in_vals)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            if hasattr(eqn.primitive, "must_handle"):
                args = subfuns + in_vals
                # This definition "reifies" the remainder of the evaluation
                # loop so it can be explicitly passed to the handler.
                def continuation(args):
                    return eval_jaxpr_loop(eqns[1:], env, eqn.outvars, [args])

                for handler in handler_stack:
                    if eqn.primitive == handler.handles:
                        return [handler.callable(continuation, *args)]
                raise ValueError("Failed to find handler.")
            else:
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


CT = jax.jit

#####
##### Primitives
#####

# Here's an example randomness primitive.
bernoulli_p = core.Primitive("bernoulli")


def bernoulli_abstract_eval(addr, p):
    return abstract_arrays.ShapedArray(p.shape, dtype=bool)


bernoulli_p.def_abstract_eval(bernoulli_abstract_eval)
bernoulli_p.must_handle = True


class Bernoulli:
    prim: core.Primitive

    def __init__(self):
        self.prim = bernoulli_p


bernoulli = Bernoulli()

# Trace primitive.
trace_p = core.Primitive("trace")


# Trace forwards bind to the primitive. This allows us to avoid encoding
# primitives as objects that JAX understands (e.g. simple numbers, or arrays)
# and allows us to also forward abstract evaluation to distribution primitive
# definition.
def trace(addr, prim, *args):
    return prim.prim.bind(addr, *args)


# Again, forward to the primitive.
def trace_abstract_eval(addr, prim, *args):
    return prim.prim.abstract_eval(addr, *args)


trace_p.def_abstract_eval(trace_abstract_eval)
trace_p.must_handle = True

# In essence, `trace` is just syntactic sugar, and just desugars
# to the primitive passed into the trace.


#####
##### Example (`simulate` as a handler)
#####

# Declare a handler + wrap in `Handler`.
def _handle_bernoulli(f, addr, p):
    key = jax.random.PRNGKey(131459)
    v = random.bernoulli(key, p)
    return f(v)


handle_bernoulli = Handler(bernoulli_p, _handle_bernoulli)

# Here's a program with our primitives.
def f(x):
    u = trace(0, bernoulli, x)
    return u


# This is the normal interpreter.
# This desugars `trace` to the underlying distribution primitives here.
print(T(0.7)(f))

# Here, we use the effect interpreter after desugaring
# `trace`, and then we stage that out to `Jaxpr` syntax.
# As expected, the handler has been eliminated!
print(T(0.7)(I_prime([handle_bernoulli], T(0.7)(f))))
