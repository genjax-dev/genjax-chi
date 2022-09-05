#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax

@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    return key, x

key = jax.random.PRNGKey(314159)
key, tr = genjax.simulate(model)(key, ())
print(tr)

