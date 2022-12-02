#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import jax.numpy as jnp
import genjax

@genjax.gen
def add_normal_noise(key, x):
    key, noise1 = genjax.trace("noise1", genjax.Normal)(
            key, 0.0, 1.0
    )
    key, noise2 = genjax.trace("noise2", genjax.Normal)(
            key, 0.0, 1.0
    )
    return (key, x + noise1 + noise2)

mapped = genjax.MapCombinator(add_normal_noise, in_axes=(0, 0))

arr = jnp.ones(100)
key = jax.random.PRNGKey(314159)
key, *subkeys = jax.random.split(key, 101)
subkeys = jnp.array(subkeys)
_, tr = jax.jit(genjax.simulate(mapped))(subkeys, (arr, ))
print(tr)

