import jax

import genjax


console = genjax.go_pretty()


@genjax.gen
def h1(key, x):
    key, m0 = genjax.trace("m0", genjax.Bernoulli)(key, (x,))
    key, m1 = genjax.trace("m1", genjax.Bernoulli)(key, (x,))
    key, m2 = genjax.trace("m2", genjax.Bernoulli)(key, (x,))
    key, m4 = genjax.trace("m3", genjax.Bernoulli)(key, (x,))
    key, m3 = genjax.trace("m4", genjax.Bernoulli)(key, (x,))
    key, m5 = genjax.trace("m5", genjax.Bernoulli)(key, (x,))
    return (key,)


@genjax.gen
def h2(key, x):
    key, m10 = genjax.trace("m10", genjax.Normal)(key, (0.0, 1.0))
    key, m11 = genjax.trace("m11", genjax.Normal)(key, (0.0, 1.0))
    key, m12 = genjax.trace("m12", genjax.Normal)(key, (0.0, 1.0))
    key, m13 = genjax.trace("m13", genjax.Normal)(key, (0.0, 1.0))
    key, m14 = genjax.trace("m14", genjax.Normal)(key, (0.0, 1.0))
    return (key,)


sw = genjax.SwitchCombinator([h1, h2])

key = jax.random.PRNGKey(314159)
key, tr = genjax.simulate(sw)(key, (1, 0.3))
console.print(tr)

chm = genjax.ChoiceMap.new({("m12",): 2.0})
fn = genjax.update(sw)
key, (w, tr, d) = fn(key, tr, chm, (1, 0.6))
