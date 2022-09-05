#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Eight schools in idiomatic GenJAX.

import jax
import jax.numpy as jnp
import genjax


# Create a MapCombinator generative function, mapping over
# the key and sigma arguments.
@genjax.gen(genjax.MapCombinator, in_axes=(0, None, None, 0))
def plate(key, mu, tau, sigma):
    key, theta = genjax.trace("theta", genjax.Normal)(key, (mu, tau))
    key, obs = genjax.trace("obs", genjax.Normal)(key, (theta, sigma))
    return key, obs

@genjax.gen
def J_schools(key, J, sigma):
    key, mu = genjax.trace("mu", genjax.Normal)(key, (0.0, 5.0))
    key, tau = genjax.trace("tau", genjax.Cauchy)(key, ())
    key, *subkeys = jax.random.split(key, J + 1)
    subkeys = jnp.array(subkeys)
    _, obs = genjax.trace("plate", plate)(subkeys, (mu, tau, sigma))
    return key, obs


# If one ever needs to specialize on arguments, you can just
# pass a lambda which closes over constants into
# a `JAXGenerativeFunction`.
#
# Here, we specialize on the number of schools.
eight_schools = genjax.JAXGenerativeFunction(
    lambda key, sigma: J_schools(key, 8, sigma)
)

