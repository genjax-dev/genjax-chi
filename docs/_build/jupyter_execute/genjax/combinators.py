#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax

@genjax.gen
def branch_1(key):
    key, x = genjax.trace("x1", genjax.Normal)(key, (0.0, 1.0))
    return (key, )

@genjax.gen
def branch_2(key):
    key, x = genjax.trace("x2", genjax.Bernoulli)(key, (0.3, ))
    return (key, )

switch = genjax.SwitchCombinator(branch_1, branch_2)

key = jax.random.PRNGKey(314159)
jitted = jax.jit(genjax.simulate(switch))
key, _ = jitted(key, (0, ))
key, tr = jitted(key, (1, ))
print(tr)


# In[2]:


import jax
import genjax

@genjax.gen
def random_walk(key, prev):
    key, x = genjax.trace("x", genjax.Normal)(key, (prev, 1.0))
    return (key, x)


unfold = genjax.UnfoldCombinator(random_walk, 1000)
init = 0.5
key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(unfold))(key, (init,))
print(tr)


# In[3]:


import jax
import jax.numpy as jnp
import genjax

@genjax.gen
def add_normal_noise(key, x):
    key, noise1 = genjax.trace("noise1", genjax.Normal)(
            key, (0.0, 1.0)
    )
    key, noise2 = genjax.trace("noise2", genjax.Normal)(
            key, (0.0, 1.0)
    )
    return (key, x + noise1 + noise2)

mapped = genjax.MapCombinator(add_normal_noise, in_axes=(0, 0))

arr = jnp.ones(100)
key = jax.random.PRNGKey(314159)
key, *subkeys = jax.random.split(key, 101)
subkeys = jnp.array(subkeys)
_, tr = jax.jit(genjax.simulate(mapped))(subkeys, (arr, ))
print(tr)

