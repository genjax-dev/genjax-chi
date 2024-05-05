# %%

import genjax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from genjax import static_gen_fn
import genjax.studio.plot as plot

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
import genjax.studio.plot as plot

data = jnp.arange(0, 10, 0.5)
key, sub_key = jax.random.split(key)

traces = jax.vmap(lambda k: model.simulate(k, (data,)))(jax.random.split(key, 10))

plot.small_multiples(
    [plot.scatter(data, ys) for ys in plot.get_address(traces, ["ys", "$ALL", "y", "value"])]
)



# %%
