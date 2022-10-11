import jax

import genjax


console = genjax.go_pretty()


@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    key, y = genjax.trace("y", genjax.Normal)(key, (x, 1.0))
    return key, y


key = jax.random.PRNGKey(314159)
obs = genjax.ChoiceMap.new({("y",): 2.0})
console.print(obs)
key, (_, tr) = model.importance(key, obs, ())
console.print(tr)

target = genjax.Selection([("x",)])
k = jax.jit(genjax.mala(target, 0.5))
key, tr = k(key, tr)
