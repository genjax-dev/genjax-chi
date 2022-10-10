#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import optax

import genjax


# Here, we can pass params as a keyword argument to
# genjax.Trainable.
#
# The combinator expects a generative function which accepts the params
# argument at the last argument position.
@genjax.gen(
    genjax.Trainable,
    params={"x": 0.5},
)
def model(key, params):
    x = params["x"]
    key, y = genjax.trace("y", genjax.Normal)(key, (x, 0.5))
    return key, y


def learning(key, lr, chm):
    optim = optax.adam(lr)
    opt_state = optim.init(model.params)
    for _ in range(0, 20):
        key, (w, tr) = genjax.importance(model)(key, chm, ())

        # Usage here.
        key, grad = model.param_grad(key, tr, scale=w)
        updates, opt_state = optim.update(grad, opt_state)
        model.update_params(updates)
    return model.params


key = jax.random.PRNGKey(314159)
learning_rate = 3e-3
obs = genjax.ChoiceMap.new({("y",): 0.2})
trained = jax.jit(learning)(key, learning_rate, obs)
