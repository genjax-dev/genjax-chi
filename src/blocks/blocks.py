import math
from dataclasses import dataclass

import genjax
import jax.numpy as jnp
import jax.random
from genjax._src.generative_functions.static.static_gen_fn import (
    StaticGenerativeFunction,
)
from genjax._src.generative_functions.supports_callees import (
    SugaredGenerativeFunctionCall as Traceable,
)
from genjax.typing import Any, Callable, Float, FloatArray, List, PRNGKey
from genjax import Pytree

RealFunction = Callable[[Float], Float]

class Block:
    gf: StaticGenerativeFunction

    def sample(self, k: PRNGKey):
        raise NotImplementedError

    def to_function(self, params: FloatArray) -> RealFunction:
        raise NotImplementedError

    def __add__(self, b: "Block"):
        return Sum(self, b)

    def __mul__(self, b: "Block"):
        return Product(self, b)

    def __matmul__(self, b: "Block"):
        return Compose(self, b)



class TracedFunction:
    def __init__(self, tr: genjax.Trace, f: Callable):
        self.tr = tr
        self.f = f

    def __call__(self, x):
        return self.f(x)

class CoinToss(Block):
    probability: float
    branches: List[Block]

    def __init__(self, probability: float, heads: Block, tails: Block):
        self.probability = probability
        self.branches = [heads, tails]
        swc = genjax.switch_combinator(heads.gf, tails.gf)
        @genjax.static_gen_fn
        def coin_toss_gf():
            a = jnp.array(genjax.flip(self.probability) @ "coin", dtype=int)
            return swc(a) @ "toss"
        self.gf = coin_toss_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    def sample(self, k: PRNGKey):
        tr = self.jitted_sample(k, ())
        ix = tr.get_choices()['toss'].index
        inner_tr = tr.get_subtrace('toss').get_subtrace(ix)
        return jax.jit(self.branches[ix].to_function(inner_tr))


class Polynomial(Block):
    def __init__(self, *, max_degree: int, coefficient_d: Traceable):
        @genjax.repeat_combinator(repeats=max_degree+1)
        @genjax.static_gen_fn
        def coefficient_gf():
            return coefficient_d @ "a"
        self.gf = coefficient_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    def to_function(self, tr: genjax.Trace):
        @jax.jit
        def f(x):
            return jax.numpy.polyval(tr.get_retval(), x)
        return TracedFunction(tr, f)

    def sample(self, k: PRNGKey):
        tr = self.jitted_sample(k, ())
        return self.to_function(tr)


class Periodic(Block):
    jitted_sample: Callable
    def __init__(self, *, amplitude: Traceable, phase: Traceable, period: Traceable):
        @genjax.static_gen_fn
        def periodic_gf():
            a = amplitude @ 'a'
            φ = phase @ 'φ'
            p = period @ 'p'
            return a, φ, p
        self.gf = periodic_gf
        self.jitted_sample = jax.jit(self.gf.simulate)


    def sample(self, k: PRNGKey) -> TracedFunction:
        tr = self.jitted_sample(k, ())
        a, φ, p = tr.get_retval()
        @jax.jit
        def f(x):
            return a * jnp.sin(φ + 2 * x * math.pi / p)
        return TracedFunction(tr, f)

class Exponential(Block):
    jitted_sample: Callable
    def __init__(self, *, a: Traceable, b: Traceable):
        @genjax.static_gen_fn
        def exponential_gf():
            a_ = a @ 'a'
            b_ = b @ 'b'
            return (a_, b_)
        self.gf = exponential_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    def sample(self, k: PRNGKey) -> TracedFunction:
        tr = self.jitted_sample(k, ())
        a, b = tr.get_retval()
        @jax.jit
        def f(x):
            return a * jnp.exp(b * x)
        return TracedFunction(tr, f)

class Compose(Block):
    f: Block
    g: Block
    def __init__(self, f: Block, g: Block):
        self.f = f
        self.g = g

    def sample(self, k: PRNGKey):
        k1, k2 = jax.random.split(k)
        f = self.f.sample(k1)
        g = self.g.sample(k2)
        @jax.jit
        def h(x):
            return f(g(x))
        return h

class PointwiseOperation(Block):
    # NB: These are not commutative, even if the underlying binary operation is,
    # due to the way randomness is threaded through the operands.
    f: Block
    g: Block

    class Trace(genjax.Trace):
        inner_f: genjax.Trace
        inner_g: genjax.Trace
        def __init__(self, f_trace, g_trace):
            self.inner_f = f_trace
            self.inner_g = g_trace

        def get_args(self) -> tuple:
            return self.inner_f.get_args(), self.inner_g.get_args()

        def get_gen_fn(self) -> genjax.GenerativeFunction:
            raise NotImplementedError

        def get_choices(self) -> genjax.ChoiceMap:
            return genjax.HierarchicalChoiceMap(genjax.Trie({
                'l': self.inner_f.get_choices().strip(),
                'r': self.inner_g.get_choices().strip()
            })).strip()

        def get_retval(self) -> tuple:
            return self.inner_f.get_retval(), self.inner_g.get_retval()

        def get_score(self) -> genjax.FloatArray:
            return jnp.add(self.inner_f.get_score(), self.inner_g.get_score())

        def project(self, s: genjax.Selection):
            raise NotImplementedError

    def __init__(self, f: Block, g: Block):
        self.f = f
        self.g = g
        # where we left off: can we make a real GF here? so that we don't need a
        # synthetic trace?


        # This is clearly the way to do it, but we have made a couple of wrong
        # turns here. The GFs of the constituent functions are not returning
        # things of the same shape. We need to take an x into the GFs representing
        # functions. The retval and the traces can be different. Need to think
        # a little harder about this.

        # Now, we already figured out that a Block GF cannot return a Python function
        # since JAX doesn't like that kind of return value.

        # The secret is going to have to involve having to_function work on
        # retvals directly!

        # So to_function is going to have to move up to the level of Block itself
        # which makes sense
        @genjax.static_gen_fn
        def pointwise_op():
            l = self.f.to_function(self.f.gf()) @ "l"
            print('L=', l)
            r = self.g.gf() @ "r"
            print('R=', r)
            return self.op(l, r)
        self.gf = pointwise_op

    def op(self, u, v):
        raise NotImplementedError

    def sample(self, k: PRNGKey):
        k1, k2 = jax.random.split(k)
        f = self.f.sample(k1)
        g = self.g.sample(k2)
        @jax.jit
        def h(x):
            return self.op(f(x), g(x))
        return TracedFunction(PointwiseOperation.Trace(f.tr, g.tr), h)


class Sum(PointwiseOperation):
    def op(self, u, v):
        return u + v

class Product(PointwiseOperation):
    def op(self, u, v):
        return u * v

def Run(b: Block, k0: PRNGKey = jax.random.PRNGKey(0)):
    k = [k0]
    while True:
        k0, k1 = jax.random.split(k[0])
        k[0] = k0
        yield b.sample(k1)

class Curve:
    model: Block
    outlier_distribution: genjax.JAXGenerativeFunction

    def __init__(self, *, model: Block, outlier_distribution: genjax.JAXGenerativeFunction):
        self.model = model
        self.outlier_distribution = outlier_distribution
        @genjax.static_gen_fn
        def model(xs, ys):
            pass
