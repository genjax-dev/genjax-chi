import jax
import genjax

# A short example with some higher-order abilities.


def g(key, x):
    key, m1 = genjax.trace("m0", genjax.Bernoulli)(key, x)
    return (key, m1)


def f(key, x):
    def _inner(key):
        key, v = g(key, x)
        key, v2 = genjax.trace("m1", genjax.Bernoulli)(key, x)
        return key, v + v2

    return key, _inner


def toplevel(key):
    key, fn = f(key, 0.3)
    key, q = genjax.trace("higher-order", fn)(key)
    return key, q


# Initialize a PRNG.
key = jax.random.PRNGKey(314159)

expr = genjax.lift(genjax.simulate(toplevel), key)
print(expr)
