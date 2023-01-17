#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax

import genjax


@genjax.gen
def model():
    x = genjax.trace("x", genjax.Normal)(0.0, 1.0)
    return x


print(model)


# In[2]:


key = jax.random.PRNGKey(314159)
jaxpr = jax.make_jaxpr(model)(key)
print(jaxpr.pretty_print(use_color=False))


# In[3]:


key = jax.random.PRNGKey(314159)
jaxpr = jax.make_jaxpr(genjax.simulate(model))(key, ())
print(jaxpr.pretty_print(use_color=False))
