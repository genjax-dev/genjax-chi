#!/usr/bin/env python
# coding: utf-8

# In[1]:


import genjax


fn = genjax.simulate(genjax.Normal)
print(fn)


# In[2]:


import jax

import genjax


@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    return key, x


key = jax.random.PRNGKey(314159)

# Usage here.
key, tr = genjax.simulate(model)(key, ())

print(tr)


# In[3]:


import jax

import genjax


@genjax.gen
def model(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    return key, x


key = jax.random.PRNGKey(314159)

# Usage here.
trace_type = genjax.get_trace_type(model)(key, ())

print(trace_type)
