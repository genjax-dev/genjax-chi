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
# a `Callable` with signature `def fn(continuation, *args)`
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
                def continuation(*args):
                    return eval_jaxpr_loop(eqns[1:], env, eqn.outvars, [*args])

                for handler in reversed(handler_stack):
                    if eqn.primitive in handler.handles:
                        try:
                            callable = getattr(handler, repr(eqn.primitive))
                            return callable(continuation, *args)
                        except:
                            return handler.callable(continuation, *args)

            ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
            if not eqn.primitive.multiple_results:
                ans = [ans]
            return eval_jaxpr_loop(eqns[1:], env, eqn.outvars, ans)
        else:
            return map(read, jaxpr.outvars)

    return eval_jaxpr_loop(jaxpr.eqns, env, jaxpr.invars, args)


#####
##### Primitives
#####

# Here's an example randomness primitive.
bernoulli_p = core.Primitive("bernoulli")


def bernoulli_abstract_eval(addr, p):
    return abstract_arrays.ShapedArray(p.shape, dtype=bool)


bernoulli_p.def_abstract_eval(bernoulli_abstract_eval)
bernoulli_p.must_handle = True

# Wrap it in a small class.
class Bernoulli:
    prim: core.Primitive

    def __init__(self):
        self.prim = bernoulli_p


bernoulli = Bernoulli()

# Trace primitive.
trace_p = core.Primitive("trace")


# `trace` just forwards `bind` (see: JAX's Autodidax for more info on `bind`)
# to the primitive. This allows us to avoid encoding
# primitives as objects that JAX understands (e.g. simple numbers, or arrays)
# and allows us to also forward abstract evaluation to distribution primitive
# definition.
def trace(addr, prim, *args):
    try:
        return prim.prim.bind(addr, *args)
    except:
        splice_p.bind(addr)
        ret = prim(*args)
        unsplice_p.bind(addr)
        return ret


splice_p = core.Primitive("splice")


def splice_abstract_eval(addr):
    return abstract_arrays.ShapedArray(shape=(0,), dtype=bool)


splice_p.def_abstract_eval(splice_abstract_eval)
splice_p.must_handle = True


unsplice_p = core.Primitive("unsplice")


def unsplice_abstract_eval(addr):
    return abstract_arrays.ShapedArray(shape=(0,), dtype=bool)


unsplice_p.def_abstract_eval(unsplice_abstract_eval)
unsplice_p.must_handle = True


def splice(addr):
    splice_p.bind


# Again, forward to the primitive.
def trace_abstract_eval(addr, prim, *args):
    return prim.prim.abstract_eval(addr, *args)


trace_p.def_abstract_eval(trace_abstract_eval)
trace_p.must_handle = True

# In essence, `trace` is just syntactic sugar, and just desugars
# to the primitive passed into the trace.

state_p = core.Primitive("state")


def state(addr, v, score):
    return state_p.bind(addr, v, score)


# Again, forward to the primitive.
def state_abstract_eval(addr, v, score):
    return v


state_p.def_abstract_eval(state_abstract_eval)
state_p.must_handle = True

seed_p = core.Primitive("seed")


def seed():
    return seed_p.bind()


def key_abstract_eval():
    return abstract_arrays.ShapedArray(shape=(2,), dtype=jnp.uint32)


seed_p.def_abstract_eval(key_abstract_eval)
seed_p.must_handle = True

#####
##### Stateful handlers
#####

# Note: both of these handlers _do not manipulate runtime values_ --
# the pointers they hold to objects like `v` and `score` are JAX `Tracer`
# values. When we do computations with these values,
# it adds to the `Jaxpr` trace.
#
# So the trick is write `callable` to coerce the return of the `Jaxpr`
# to send out the accumulated state we want.

# Handles `seed` -- a.k.a. give me a new split random key.
class PRNGProvider(Handler):
    def __init__(self, seed: int):
        self.handles = [seed_p]
        self.state = jax.random.PRNGKey(seed)

    def callable(self, f):
        key, sub_key = jax.random.split(self.state)
        self.state = key
        return f(sub_key)


# Handles `state` -- a.k.a. I want to lift `return a :: A` into `M A`.
class TraceRecorder(Handler):
    def __init__(self):
        self.handles = [state_p, splice_p, unsplice_p]
        self.state = {}
        self.level = []
        self.score = 0.0
        self.return_or_continue = False

    def callable(self, f, addr, v, score):
        self.state[(*self.level, addr)] = v
        self.score += score
        if self.return_or_continue:
            return f(v)
        else:
            self.return_or_continue = True
            ret = f(v)
            return (ret, self.state, self.score)

    def splice(self, f, addr):
        self.level.append(addr)
        return f([])

    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f([])
        except:
            return f([])


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


#####
##### Example
#####

# Declare a handler + wrap in `Handler`.
def _handle_bernoulli(f, addr, p):
    key = seed()
    v = random.bernoulli(key, p)
    score = stats.bernoulli.logpmf(v, p)
    state(addr, v, score)
    return f(v)


handle_bernoulli = Handler([bernoulli_p], _handle_bernoulli)


# Here's a hierarchical program with our primitives.
def g1():
    q1 = trace(0, bernoulli, 0.5)
    q2 = trace(1, bernoulli, 0.5)
    return q1 + q2


def g2():
    q = trace(0, bernoulli, 0.5)
    return q


def f(x):
    m1 = trace(0, g1)
    m2 = trace(1, g2)
    return 2 * (m1 + m2)


# This is the normal JAX tracer.
# This desugars `trace` to the underlying distribution primitives here.
print(T(0.2)(f))

# Here, we use the effect interpreter after desugaring
# `trace`, and then we stage that out to `Jaxpr` syntax.
expr = lift(f, 0.2)
# This eliminates `bernoulli`, but raises `seed` and `state` --
# asking for a PRNG and a place to put the choice values + scores.
expr = handle([handle_bernoulli], expr)
expr = lift(expr, 0.2)

# To provide PRNG seeds, plus a place to put choice values + scores,
# we provide two stateful handlers.
#
# "Stateful" is a misnomer here -- these handlers hold `Tracer` values for JAX,
# and are convenient containers to guide code generation.
p = PRNGProvider(50)
r = TraceRecorder()
expr = handle([r, p], expr)
v = jax.jit(expr)(0.2)
print(v)
