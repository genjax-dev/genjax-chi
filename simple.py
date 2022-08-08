#####
##### Example
#####

import jax
import gex


def f(key, x):
    key, m1 = gex.trace("m1", gex.Bernoulli)(key, x)
    key, m2 = gex.trace("m2", gex.Bernoulli)(key, x)
    return (key, 2 * (m1 + m2))


key = jax.random.PRNGKey(314159)
expr = gex.lift(f, key, 0.3)
print(expr)
fn = gex.Simulate().jit(f)(key, 0.3)
tr = fn(key, 0.3)
print(tr.get_choices())

chm = {("m1",): True}
fn = gex.Generate(chm).jit(f)(key, 0.3)
w, tr = fn(key, 0.3)
print((w, tr.get_choices()))
