#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax
console = genjax.pretty()

@genjax.gen
def random_walk(prev):
    x = genjax.trace("x", genjax.Normal)(prev, 1.0)
    return x


unfold = genjax.Unfold(random_walk, 1000)
init = 0.5
key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(unfold))(key, (999, init))
console.print(tr)

