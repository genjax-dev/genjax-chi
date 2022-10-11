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


console = genjax.go_pretty()

# An example of combining multiple combinators to form
# larger patterns of generative computation.


@genjax.gen(genjax.Map, in_axes=(0,))
def model1(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    key, q = genjax.trace("y", genjax.Bernoulli)(key, (0.3,))
    return (key,)


@genjax.gen(genjax.Map, in_axes=(0,))
def model2(key):
    key, x = genjax.trace("y", genjax.Bernoulli)(key, (0.3,))
    return (key,)


@genjax.gen(genjax.Map, in_axes=(0,))
def model3(key):
    key, x = genjax.trace("z", genjax.Normal)(key, (0.5, 1.0))
    key, y = genjax.trace("m", genjax.Normal)(key, (0.3, 2.0))
    return (key,)


sw = genjax.Switch([model1, model2, model3])


def fn():
    key = jax.random.PRNGKey(314159)
    key, *sub_keys = jax.random.split(key, 6)
    sub_keys = jnp.array(sub_keys)
    _, tr = genjax.simulate(sw)(sub_keys, (1,))
    key, *sub_keys = jax.random.split(key, 6)
    sub_keys = jnp.array(sub_keys)
    chm = genjax.VectorChoiceMap.new(
        np.array([i for i in range(0, 5)]),
        genjax.ChoiceMap.new({("z",): np.array([0.5 for _ in range(0, 5)])}),
    )
    _, (w, new, d) = jax.jit(genjax.update(sw))(
        sub_keys,
        tr,
        chm,
        (2,),
    )
    return w, new


w, new = jax.jit(fn)()
console.print(new)
console.print(w)
