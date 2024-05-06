import math
from typing import override

import genjax
import jax.numpy as jnp
import jax.random
from genjax import Pytree
from genjax._src.core.typing import ArrayLike
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

    def sample(self, n: int = 1, k: PRNGKey = jax.random.PRNGKey(0)):
        return jax.vmap(self.jitted_sample, in_axes=(0, None))(jax.random.split(k, n), ())

    def __add__(self, b: "Block"):
        return Pointwise(self, b, lambda a, b: jnp.add(a, b))

    def __mul__(self, b: "Block"):
        return Pointwise(self, b, lambda a, b: jnp.multiply(a, b))

    def __matmul__(self, b: "Block"):
        return Compose(self, b)


class BlockFunction(Pytree):
    """A BlockFunction is a Pytree which is also Callable."""

    def __call__(self, x: ArrayLike) -> FloatArray:
        raise NotImplementedError


class BinaryOperation(BlockFunction):
    l: BlockFunction
    r: BlockFunction
    op: Callable[[ArrayLike, ArrayLike], FloatArray] = Pytree.static()

    def __call__(self, x: ArrayLike) -> FloatArray:
        return self.op(self.l(x), self.r(x))


class Polynomial(Block):
    def __init__(self, *, max_degree: int, coefficient_d: Traceable):
        @genjax.repeat_combinator(repeats=max_degree + 1)
        @genjax.static_gen_fn
        def coefficient_gf() -> FloatArray:
            return coefficient_d @ "coefficients"

        @genjax.static_gen_fn
        def polynomial_gf() -> BlockFunction:
            return Polynomial.Function(coefficient_gf() @ "p")

        self.gf = polynomial_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    class Function(BlockFunction):
        coefficients: FloatArray
        @override
        def __call__(self, x: FloatArray):
            deg = self.coefficients.shape[-1]
            powers = jnp.pow(jnp.broadcast_to(x, deg), jax.lax.iota(dtype=int, size=deg))
            return jax.numpy.matmul(self.coefficients, powers)

class Periodic(Block):
    def __init__(self, *, amplitude: Traceable, phase: Traceable, period: Traceable):
        @genjax.static_gen_fn
        def periodic_gf() -> BlockFunction:
            return Periodic.Function(
                amplitude @ "a",
                phase @ "φ",
                period @ "T"
            )

        self.gf = periodic_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    class Function(BlockFunction):
        amplitude: FloatArray
        phase: FloatArray
        period: FloatArray
        @override
        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.amplitude * jnp.sin(self.phase + 2 * x * math.pi / self.period)


class Exponential(Block):
    def __init__(self, *, a: Traceable, b: Traceable):
        @genjax.static_gen_fn
        def exponential_gf() -> BlockFunction:
            return Exponential.Function(a @ "a", b @ "b")

        self.gf = exponential_gf
        self.jitted_sample = jax.jit(self.gf.simulate)

    class Function(BlockFunction):
        a: FloatArray
        b: FloatArray
        @override
        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.a * jnp.exp(self.b * x)


class Pointwise(Block):
    # NB: These are not commutative, even if the underlying binary operation is,
    # due to the way randomness is threaded through the operands.
    def __init__(self, f: Block, g: Block, op: Callable[[ArrayLike, ArrayLike], FloatArray]):
        self.f = f
        self.g = g

        @genjax.static_gen_fn
        def pointwise_op() -> BlockFunction:
            return BinaryOperation(f.gf() @ "l", g.gf() @ "r", op)

        self.gf = pointwise_op
        self.jitted_sample = jax.jit(self.gf.simulate)


class Compose(Block):
    def __init__(self, f: Block, g: Block):
        @genjax.static_gen_fn
        def composition() -> BlockFunction:
            return Compose.Function(f.gf() @ "l", g.gf() @ "r")

        self.gf = composition
        self.jitted_sample = jax.jit(
            self.gf.simulate
        )  # TODO: move this copypasta to post_init?

    class Function(BlockFunction):
        f: BlockFunction = Pytree.field()
        g: BlockFunction = Pytree.field()

        @override
        def __call__(self, x: ArrayLike) -> FloatArray:
            return self.f(self.g(x))


class CoinToss(Block):
    def __init__(self, probability: float, heads: Block, tails: Block):
        swc = genjax.switch_combinator(tails.gf, heads.gf)

        @genjax.static_gen_fn
        def coin_toss_gf() -> StaticGenerativeFunction:
            a = jnp.array(genjax.flip(probability) @ "coin", dtype=int)
            choice = swc(a) @ "toss"
            return choice

        self.gf = coin_toss_gf
        self.jitted_sample = jax.jit(self.gf.simulate)


class CurveFit:
    curve: Block
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
        def kernel(x: ArrayLike, f: Callable[[ArrayLike], FloatArray]) -> StaticGenerativeFunction:
            is_outlier = genjax.flip(0.2) @ 'outlier'
            io = jnp.array(is_outlier, dtype = int)
            return swc(io, f(x)) @ 'y'

        @genjax.static_gen_fn
        def model(xs: FloatArray) -> FloatArray:
            c = curve.gf() @ 'curve'
            ys = kernel(xs, c) @ 'ys'
            return ys

        self.gf = model
        self.jitted_importance = jax.jit(self.gf.importance)

    def importance_sample(self, xs: FloatArray, ys: FloatArray, N: int, K: int, key: PRNGKey = jax.random.PRNGKey(0)):
        choose_ys = genjax.choice_map({
            'ys': genjax.vector_choice_map(
                genjax.choice_map({
                    ('y', 'value'): ys
                })
            )
        })
        k1, k2 = jax.random.split(key)
        trs, ws = jax.vmap(self.jitted_importance, in_axes=(0, None, None))(
            jax.random.split(k1, N),
            choose_ys,
            (xs,)
        )
        ixs = jax.vmap(jax.jit(genjax.categorical.sample), in_axes=(0, None))(
            jax.random.split(k2, K),
            ws
        )

        curves = trs.get_subtrace('curve').get_retval()
        return jax.tree.map(lambda x: x[ixs], curves)
