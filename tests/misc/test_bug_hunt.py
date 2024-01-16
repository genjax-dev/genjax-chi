import genjax
import jax
import jax.numpy as jnp
from genjax import interpreted as gen


class TestBugHunt:

    @staticmethod
    def polynomial(coefficients):
        """Given coefficients of a polynomial a_0, a_1, ..., return a function
        computing a_0 x^0 + a_1 x^1 + ..."""

        def f(x):
            powers_of_x = jnp.array([x**i for i in range(len(coefficients))])
            return jnp.sum(coefficients * powers_of_x)

        return f


    @gen
    @staticmethod
    def model_y(y, scale):
        y = genjax.normal(y, scale) @ "value"
        return y

    @gen
    @staticmethod
    def kernel(xs, f):
        y = []
        for i, x in enumerate(xs):
            is_outlier = genjax.flip(0.1) @ ("outlier", i)
            scale = 30.0 if is_outlier else 0.3
            y.append(TestBugHunt.model_y(f(x), scale) @ ("y", i))

        return jnp.array(y)

    @gen
    @staticmethod
    def model(xs):
        coefficients = genjax.mv_normal(jnp.zeros(3), 2.0 * jnp.identity(3)) @ "alpha"
        f = TestBugHunt.polynomial(coefficients)
        ys = TestBugHunt.kernel(xs, f) @ "ys"
        return ys

    def test1(self):
        key = jax.random.PRNGKey(0)

        xs = jnp.arange(0, 10, 0.5)

        key, sub_key = jax.random.split(key)
        tr = self.model.simulate(sub_key, (xs,))

        print(tr.get_retval())
        assert True
