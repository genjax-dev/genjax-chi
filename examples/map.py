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


@genjax.gen
def add_normal_noise(key, x):
    key, noise = genjax.trace("noise", genjax.Normal)(key, ())
    return (key, x + noise)


mapped = genjax.MapCombinator(add_normal_noise)

key = jax.random.PRNGKey(314159)
key, *subkeys = jax.random.split(key, 3)
subkeys = jnp.array(subkeys)
key, tr = jax.jit(genjax.simulate(mapped))(
    subkeys,
    (jnp.array([0.3, 0.8]),),
)
print(tr.get_retval())
