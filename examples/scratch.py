import jax

import genjax


console = genjax.go_pretty()


@genjax.gen
def h(key, x):
    key, m1 = genjax.trace("m0", genjax.Bernoulli)(key, (x,))
    return (key, m1)


@genjax.gen
def g(key, x):
    key, m1 = genjax.trace("m0", h)(key, (x,))
    return (key, m1)


@genjax.gen
def f(key, x):
    key, m0 = genjax.trace("m0", genjax.Bernoulli, shape=(3, 3))(key, (x,))
    key, m4 = genjax.trace("m4", g)(key, (x,))
    key, m5 = genjax.trace("m5", genjax.Normal)(key, (0.0, 1.0))
    return key, m0 * m4


key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(f))(key, (0.3,))
console.print(tr)
