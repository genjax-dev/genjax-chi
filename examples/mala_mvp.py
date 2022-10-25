import jax

import genjax


console = genjax.go_pretty()


@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    key, y = genjax.trace("y", genjax.Normal)(key, (x, 0.1))
    return key, y


key = jax.random.PRNGKey(314159)
target = genjax.Selection([("x",)])
obs = genjax.ChoiceMap.new({("y",): 3.0})
key, (_, tr) = model.importance(key, obs, ())

k = jax.jit(genjax.mala(target, 0.01))
for _ in range(0, 50):
    key, (tr, _) = k(key, tr)
    console.print(tr["x"])
