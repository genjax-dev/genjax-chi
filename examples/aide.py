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

"""
Showcases running AIDE on TPUs.
"""

import jax
import jax.numpy as jnp
import genjax


def f(key, p1):
    key, r = genjax.trace("p", genjax.Beta)(key, 1, 1)
    key, x = genjax.trace("x", genjax.Bernoulli)(key, r)
    key, y = genjax.trace("y", genjax.Bernoulli)(key, p1)
    return key, x + y


key = jax.random.PRNGKey(314159)
key, *subkeys = jax.random.split(key, 100 + 1)
subkeys = jnp.array(subkeys)
_, ests = jax.vmap(
    jax.jit(genjax.experimental.aide(f, f, 100, 100)),
    in_axes=(0, None, None),
)(subkeys, (0.3,), (0.8,))
print(jnp.mean(ests))
