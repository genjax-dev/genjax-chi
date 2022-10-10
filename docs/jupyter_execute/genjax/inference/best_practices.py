#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax

def bootstrap_importance_sampling(model: genjax.GenerativeFunction, n_particles: int):
    def _inner(key, model_args: Tuple, observations: genjax.ChoiceMap):
        key, *subkeys = jax.random.split(key, n_particles + 1)
        subkeys = jnp.array(subkeys)
        _, (lws, trs) = jax.vmap(model.importance, in_axes=(0, None, None))(
            subkeys,
            observations,
            model_args,
        )
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(n_particles)
        return key, (trs, log_normalized_weights, log_ml_estimate)

    return _inner

