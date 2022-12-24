#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax

@genjax.gen
def random_walk(key, prev):
    key, x = genjax.trace("x", genjax.Normal)(key, prev, 1.0)
    return (key, x)


unfold = genjax.UnfoldCombinator(random_walk, 1000)
init = 0.5
key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(unfold))(key, (999, init))
print(tr)

