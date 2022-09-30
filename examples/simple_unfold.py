import jax
import numpy as np
import genjax

console = genjax.go_pretty()


@genjax.gen(genjax.Unfold, max_length=10)
def fn(key, prev_state):
    key, new = genjax.trace("z", genjax.Normal)(key, (prev_state, 1.0))
    return key, new


obs = genjax.ChoiceMap.new({("z",): np.ones(5)})

key = jax.random.PRNGKey(314159)
key, (w, tr) = genjax.importance(fn)(key, obs, (10, 0.1))
console.print(tr.get_choices())
