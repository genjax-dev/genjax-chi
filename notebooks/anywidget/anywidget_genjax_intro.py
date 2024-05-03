# %% 

import genjax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from genjax import static_gen_fn

key = jax.random.PRNGKey(314159)

# %% 

# Two branches for a branching submodel.
@static_gen_fn
def model_y(x, coefficients):
    basis_value = jnp.array([1.0, x, x**2])
    polynomial_value = jnp.sum(basis_value * coefficients)
    y = genjax.normal(polynomial_value, 0.3) @ "value"
    return y


@static_gen_fn
def outlier_model(x, coefficients):
    basis_value = jnp.array([1.0, x, x**2])
    polynomial_value = jnp.sum(basis_value * coefficients)
    y = genjax.normal(polynomial_value, 30.0) @ "value"
    return y


# The branching submodel.
switch = genjax.switch_combinator(model_y, outlier_model)

# A mapped kernel function which calls the branching submodel.


@genjax.map_combinator(in_axes=(0, None))
@static_gen_fn
def kernel(x, coefficients):
    is_outlier = genjax.flip(0.1) @ "outlier"
    is_outlier = jnp.array(is_outlier, dtype=int)
    y = switch(is_outlier, x, coefficients) @ "y"
    return y


@static_gen_fn
def model(xs):
    coefficients = (
        genjax.mv_normal(np.zeros(3, dtype=float), 2.0 * np.identity(3)) @ "alpha"
    )
    ys = kernel(xs, coefficients) @ "ys"
    return ys

# %% 
from timeit import default_timer as timer

class benchmark(object):

    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t
        
data = jnp.arange(0, 10, 0.5)
key, sub_key = jax.random.split(key)
tr = jax.jit(model.simulate)(sub_key, (data,))
chm = tr.get_choices()
values = [chm["ys", i, "y", "value"] for i in range(len(data))]
values
[tr["ys", i, "y", "value"]  for i in range(len(data))]
tr["ys" "y"]
with benchmark("inner"):
    tr.get_choices()["ys"].inner['y']['value']
with benchmark("comprehension"):    
    [chm["ys", i, "y", "value"] for i in range(len(data))]
jitted_lambda = jax.jit(lambda v: chm["ys", v, "y", "value"])    
with benchmark("vmap"):
    jax.vmap(jitted_lambda)(jnp.arange(0, len(data)))    
tr.get_choices()["ys"].inner['y']['value']

# %% 
from pyobsplot import Plot, Obsplot
from ipywidgets import HBox
# model.simulate(key, data)
traces = jax.vmap(lambda k: model.simulate(k, (data,)))(jax.random.split(key, 5))

p = Obsplot(default={'width': 200})

def resolve_address(tr, address):
    if isinstance(address, str):
        return tr[address]
    else:
        result = tr
        for part in address:
            if part == "$ALL":
                result = result.inner
            else:
                result = result[part]
        return result

def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return x.tolist()

def scatter(x, y):
    Plot.dot({'length': len(x)}, {'x':ensure_list(x), 'y': ensure_list(y)})

HBox([Plot.plot(scatter())])
p.width = 500 
p
# %%
