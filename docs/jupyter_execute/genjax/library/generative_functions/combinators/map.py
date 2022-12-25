#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import jax.numpy as jnp
import genjax
console = genjax.pretty()

@genjax.gen
def add_normal_noise(x):
    noise1 = genjax.trace("noise1", genjax.Normal)(
            0.0, 1.0
    )
    noise2 = genjax.trace("noise2", genjax.Normal)(
            0.0, 1.0
    )
    return (key, x + noise1 + noise2)

mapped = genjax.MapCombinator.new(add_normal_noise, in_axes=(0,))

arr = jnp.ones(100)
key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(mapped))(key, (arr, ))
console.print(tr)

