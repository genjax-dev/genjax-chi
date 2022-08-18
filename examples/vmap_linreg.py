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
import genjax


def _kernel(key, alpha, beta, x):
    v = alpha * x + beta
    key, out = genjax.trace("y", genjax.Uniform)(key, v - 4.0, v + 4.0)
    return key, out


kernel = jax.vmap(_kernel, in_axes=(0, None, None, 0))


def model(key, xs):
    key, alpha = genjax.trace("alpha", genjax.Uniform)(key, 0.0, 2.0)
    key, beta = genjax.trace("beta", genjax.Uniform)(key, -3.0, 3.0)
    key, *subkeys = jax.random.split(key, 4)
    subkeys = jnp.array(subkeys)
    key, ys = genjax.trace("obs", kernel)(subkeys, alpha, beta, xs)
    return key, ys


key = jax.random.PRNGKey(314159)
ys = genjax.ChoiceMap({("obs", "y"): jnp.array([3.0, 4.0, 5.0])})
key, *subkeys = jax.random.split(key, 50 + 1)
subkeys = jnp.array(subkeys)
_, (ws, trs) = jax.jit(
    jax.vmap(genjax.importance(model), in_axes=(0, None, None))
)(subkeys, ys, (jnp.array([1.0, 2.0, 3.0]),))
v = jax.random.categorical(key, ws)
print(trs)
