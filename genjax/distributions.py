import jax
from jax._src import abstract_arrays


class Bernoulli:
    def abstract_eval(self, key, p):
        return (key, abstract_arrays.ShapedArray(shape=p.shape, dtype=bool))

    def sample(self, key, p):
        key, sub_key = jax.random.split(key)
        v = jax.random.bernoulli(key, p)
        return (key, v)

    def score(self, v, p):
        return jax.scipy.stats.bernoulli.logpmf(v, p)


class Normal:
    def abstract_eval(self, key):
        return (key, abstract_arrays.ShapedArray(shape=(1,), dtype=float))

    def sample(self, key):
        key, sub_key = jax.random.split(key)
        v = jax.random.normal(key)
        return (key, v)

    def score(self, v):
        return jax.scipy.stats.norm.logpdf(v)
