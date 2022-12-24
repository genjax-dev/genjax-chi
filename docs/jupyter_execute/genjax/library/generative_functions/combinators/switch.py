#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax

import genjax


@genjax.gen
def branch_1(key):
    key, x = genjax.trace("x1", genjax.Normal)(key, 0.0, 1.0)
    return (key,)


@genjax.gen
def branch_2(key):
    key, x = genjax.trace("x2", genjax.Bernoulli)(key, 0.3)
    return (key,)


switch = genjax.SwitchCombinator([branch_1, branch_2])

key = jax.random.PRNGKey(314159)
jitted = jax.jit(genjax.simulate(switch))
key, _ = jitted(key, (0,))
key, tr = jitted(key, (1,))
print(tr)
