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
    key, noise1 = genjax.trace("noise1", genjax.Normal)(key, ())
    key, noise2 = genjax.trace("noise2", genjax.Normal)(key, ())
    return (key, x + noise1 + noise2)


mapped = genjax.MapCombinator(add_normal_noise)

key = jax.random.PRNGKey(314159)
key, *subkeys = jax.random.split(key, 3)
arr = jnp.ones(2)
subkeys = jnp.array(subkeys)
key, tr = jax.jit(genjax.simulate(mapped))(subkeys, (arr,))
print(tr.get_retval())

chm = genjax.ChoiceMap({(1, "noise1"): 0.0})
jitted = jax.jit(genjax.importance(mapped))
key, (w, tr) = jitted(
    subkeys,
    chm,
    (arr,),
)
print(w)
