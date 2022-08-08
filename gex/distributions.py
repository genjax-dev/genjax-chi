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
