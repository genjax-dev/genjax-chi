import math

import genjax
import jax.numpy as jnp
import jax.random
from genjax import Pytree
from genjax._src.generative_functions.static.static_gen_fn import (
    StaticGenerativeFunction,
)
from genjax._src.generative_functions.supports_callees import (
    SugaredGenerativeFunctionCall as Traceable,
)
from genjax.typing import Callable, FloatArray, PRNGKey


class Block:
    gf: StaticGenerativeFunction
    jitted_sample: Callable

    def sample(self, k: PRNGKey):
        tr = self.jitted_sample(k, ())
        return tr

    def __add__(self, b: "Block"):
        return Pointwise(self, b, lambda a, b: a + b)

    def __mul__(self, b: "Block"):
        return Pointwise(self, b, lambda a, b: a * b)

    def __matmul__(self, b: "Block"):
        return Compose(self, b)


class BlockFunction(Pytree):
    """A BlockFunction is a Pytree which is also Callable."""

    def __call__(self, x: float):
        raise NotImplementedError


class BinaryOperation(BlockFunction):
    l: BlockFunction
    r: BlockFunction
    op: Callable = Pytree.static()  # TODO: refine type

    def __call__(self, x: float):
        return self.op(self.l(x), self.r(x))


class Polynomial(Block):
    def __init__(self, *, max_degree: int, coefficient_d: Traceable):
        @genjax.repeat_combinator(repeats=max_degree + 1)
        @genjax.static_gen_fn
        def coefficient_gf():
            return coefficient_d @ "coefficients"

        @genjax.static_gen_fn
        def polynomial_gf():
            return Polynomial.Function(coefficient_gf() @ "p")

        self.gf = polynomial_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    class Function(BlockFunction):
        params: FloatArray

        def __call__(self, x: float):
            return jax.numpy.polyval(self.params, jnp.array(x))


class Periodic(Block):
    def __init__(self, *, amplitude: Traceable, phase: Traceable, period: Traceable):
        @genjax.static_gen_fn
        def periodic_gf():
            return Periodic.Function(amplitude @ "a", phase @ "φ", period @ "T")

        self.gf = periodic_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    class Function(BlockFunction):
        amplitude: FloatArray
        phase: FloatArray
        period: FloatArray

        def __call__(self, x: float):
            return self.amplitude * jnp.sin(self.phase + 2 * x * math.pi / self.period)


class Exponential(Block):
    def __init__(self, *, a: Traceable, b: Traceable):
        @genjax.static_gen_fn
        def exponential_gf():
            return Exponential.Function(a @ "a", b @ "b")

        self.gf = exponential_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    class Function(BlockFunction):
        a: FloatArray
        b: FloatArray

        def __call__(self, x: float):
            return self.a * jnp.exp(self.b * x)


class Pointwise(Block):
    # NB: These are not commutative, even if the underlying binary operation is,
    # due to the way randomness is threaded through the operands.
    def __init__(self, f: Block, g: Block, op: Callable[[float, float], float]):
        self.f = f
        self.g = g

        @genjax.static_gen_fn
        def pointwise_op():
            return BinaryOperation(f.gf() @ "l", g.gf() @ "r", op)

        self.gf = pointwise_op
        self.jitted_sample = jax.jit(self.gf.simulate)


class Compose(Block):
    def __init__(self, f: Block, g: Block):
        @genjax.static_gen_fn
        def composition():
            return Compose.Function(f.gf() @ "l", g.gf() @ "r")

        self.gf = composition
        self.jitted_sample = jax.jit(
            self.gf.simulate
        )  # TODO: move this copypasta to post_init?

    class Function(BlockFunction):
        f: BlockFunction = Pytree.field()
        g: BlockFunction = Pytree.field()

        def __call__(self, x: float):
            return self.f(self.g(x))


class CoinToss(Block):
    def __init__(self, probability: float, heads: Block, tails: Block):
        swc = genjax.switch_combinator(tails.gf, heads.gf)

        @genjax.static_gen_fn
        def coin_toss_gf():
            a = jnp.array(genjax.flip(probability) @ "coin", dtype=int)
            choice = swc(a) @ "toss"
            return choice

        self.gf = coin_toss_gf
        self.jitted_sample = jax.jit(self.gf.simulate)


def Run(b: Block, k0: PRNGKey = jax.random.PRNGKey(0)):
    while True:
        k0, k1 = jax.random.split(k0)
        yield b.sample(k1)


class CurveFit:
    curve: Block
    #outlier_distribution: genjax.JAXGenerativeFunction
    xs: FloatArray
    gf: StaticGenerativeFunction
    jitted_importance: Callable

    def __init__(
        self, *, curve: Block,
        inlier_model: StaticGenerativeFunction,
        outlier_model: StaticGenerativeFunction
    ):
        self.curve = curve

        swc = genjax.switch_combinator(inlier_model, outlier_model)

        @genjax.map_combinator(in_axes=(0, None))
        @genjax.static_gen_fn
        def kernel(x, f):
            is_outlier = genjax.flip(0.2) @ 'outlier'
            io = jnp.array(is_outlier, dtype = int)
            return swc(io, f(x)) @ 'y'

        @genjax.static_gen_fn
        def model(xs):
            c = curve.gf() @ 'curve'
            ys = kernel(xs, c) @ 'ys'
            return ys

        self.gf = model
        self.jitted_importance = jax.jit(self.gf.importance)

    def importance_sample(self, xs, ys, N, K):
        choose_ys = genjax.choice_map({
            'ys': genjax.vector_choice_map(
                genjax.choice_map({
                    ('y', 'value'): ys
                })
            )
        })
        k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
        trs, ws = jax.vmap(self.jitted_importance, in_axes=(0, None, None))(
            jax.random.split(k1, N),
            choose_ys,
            (xs,)
        )
        ixs = jax.vmap(jax.jit(genjax.categorical.sample), in_axes=(0, None))(
            jax.random.split(k2, K),
            ws
        )
        return trs.get_subtrace('curve').get_retval(), ixs
