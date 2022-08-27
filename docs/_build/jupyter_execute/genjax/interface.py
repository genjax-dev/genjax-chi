#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax

@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, ())
    return key, x

key = jax.random.PRNGKey(314159)
key, x = genjax.sample(model)(key, ())
print(x)


# In[2]:


import jax
import genjax

@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, ())
    return key, x

key = jax.random.PRNGKey(314159)
key, tr = genjax.simulate(model)(key, ())
print(tr)

