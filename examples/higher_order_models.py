import jax
import genjax as gex

# A short example with some higher-order abilities.


def g(key, x):
    key, m1 = gex.trace("m0", gex.Bernoulli)(key, x)
    return (key, m1)


def f(key, x):
    def _inner(key):
        key, v = g(key, x)
        key, v2 = gex.trace("m1", gex.Bernoulli)(key, x)
        return key, v + v2

    return key, _inner


def toplevel(key):
    key, fn = f(key, 0.3)
    key, q = gex.trace("higher-order", fn)(key)
    return key, q


# Initialize a PRNG.
key = jax.random.PRNGKey(314159)

tr = gex.simulate(toplevel)(key)
print(tr.get_choices())
