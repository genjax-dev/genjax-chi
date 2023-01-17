#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax

import genjax


console = genjax.pretty()


@genjax.gen
def branch_1():
    x = genjax.trace("x1", genjax.Normal)(0.0, 1.0)


@genjax.gen
def branch_2():
    x = genjax.trace("x2", genjax.Bernoulli)(0.3)


switch = genjax.SwitchCombinator([branch_1, branch_2])

key = jax.random.PRNGKey(314159)
jitted = jax.jit(genjax.simulate(switch))
key, _ = jitted(key, (0,))
key, tr = jitted(key, (1,))
console.print(tr)
