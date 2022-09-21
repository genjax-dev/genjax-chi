#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
import genjax
import optax

@genjax.gen(
    genjax.Learnable,
    params={"x": 0.5},
)
def model(key, params):
    x = params["x"]
    key, y = genjax.trace("y", genjax.Normal)(key, (x, 0.5))
    return key, y

def learning(key, lr, chm):
    optim = optax.adam(lr)
    opt_state = optim.init(model.params)
    for _ in range(0, 100):
        key, (w, tr) = genjax.importance(model)(key, chm, ())
        key, grad = model.param_grad(key, tr, scale=w)
        updates, opt_state = optim.update(grad, opt_state)
        model.update_params(updates)
    return model.params


key = jax.random.PRNGKey(314159)
learning_rate = 3e-3
obs = genjax.ChoiceMap.new({("y",): 0.2})
trained = jax.jit(learning)(key, learning_rate, obs)

