import jax
import genjax


@genjax.gen
def h(key):
    key, x = genjax.trace("x", genjax.Normal)(key, ())
    key, y = genjax.trace("y", genjax.Normal)(key, ())
    return key, x + y


key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(h))(key, ())
print(tr)
