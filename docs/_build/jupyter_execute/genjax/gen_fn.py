#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax

@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, ())
    return key, x

print(model)


# In[2]:


key = jax.random.PRNGKey(314159)
print(jax.make_jaxpr(model)(key))


# In[3]:


key = jax.random.PRNGKey(314159)
print(jax.make_jaxpr(genjax.simulate(model))(key, ()))

