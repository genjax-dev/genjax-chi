# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import genjax

# Eight schools in idiomatic GenJAX.


# Create a MapCombinator generative function, mapping over
# the key and sigma arguments.
@genjax.gen(genjax.Map, in_axes=(0, None, None, 0))
def plate(key, mu, tau, sigma):
    key, theta = genjax.trace("theta", genjax.Normal)(key, (mu, tau))
    key, obs = genjax.trace("obs", genjax.Normal)(key, (theta, sigma))
    return key, obs


@genjax.gen
def J_schools(key, J, sigma):
    key, mu = genjax.trace("mu", genjax.Normal)(key, (0.0, 5.0))
    key, tau = genjax.trace("tau", genjax.HalfCauchy)(key, (5.0, 1.0))
    key, *subkeys = jax.random.split(key, J + 1)
    subkeys = jnp.array(subkeys)
    _, obs = genjax.trace("plate", plate)(subkeys, (mu, tau, sigma))
    return key, obs


# If one ever needs to specialize on arguments, you can just pass a lambda
# which closes over constants into a `JAXGenerativeFunction`.
#
# Here, we specialize on the number of schools.
eight_schools = genjax.gen(lambda key, sigma: J_schools(key, 8, sigma))

# Here's an implementation of jittable importance resampling, using
# the prior as a proposal.
def importance_resampling(model, key, args, obs, n_particles):
    key, *subkeys = jax.random.split(key, n_particles + 1)
    subkeys = jnp.array(subkeys)
    _, (w, tr) = jax.vmap(genjax.importance(model), in_axes=(0, None, None))(
        subkeys, obs, args
    )
    ind = jax.random.categorical(key, w)
    tr = jax.tree_util.tree_map(lambda v: v[ind], tr)
    w = w[ind]
    return key, (w, tr)


# Key, arguments, and observations.
key = jax.random.PRNGKey(314159)
sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
obs = genjax.ChoiceMap(
    {
        ("plate",): genjax.VectorChoiceMap(
            np.array([i for i in range(0, 8)]),
            {
                ("obs",): jnp.array(
                    [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
                )
            },
        )
    }
)

# Check trace type.
trace_type = genjax.get_trace_type(eight_schools)(key, (sigma,))
print(trace_type)


# We JIT -- we specialize on the number of particles so JAX's PRNG
# split function traces properly.
jitted = jax.jit(importance_resampling, static_argnums=4)
key, (w, tr) = jitted(eight_schools, key, (sigma,), obs, 1000)

inv_mass_matrix = 0.5 * np.ones(10)
step_size = 1e-3
key, states = jax.jit(
    genjax.nuts(
        tr,
        genjax.Selection([("plate", "obs")]).complement(),
        500,
        step_size,
        inv_mass_matrix,
    )
)(key)
print(states)
