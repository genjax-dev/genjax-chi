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
import genjax


@genjax.gen
def h1(key, x):
    key, m0 = genjax.trace("m0", genjax.Bernoulli)(key, x)
    key, m1 = genjax.trace("m1", genjax.Bernoulli)(key, x)
    return (key,)


@genjax.gen
def h2(key, x):
    key, m0 = genjax.trace("m0", genjax.Normal)(key)
    key, m1 = genjax.trace("m1", genjax.Normal)(key)
    return (key,)


sw = genjax.SwitchCombinator(h1, h2)


@genjax.gen
def f(key, x):
    key, m0 = genjax.trace("m0", genjax.Bernoulli)(key, x)
    (key,) = genjax.trace("m5", sw)(key, m0, x)
    return key, 2 * m0


key = jax.random.PRNGKey(314159)
key, tr = jax.jit(genjax.simulate(f))(key, (0.3,))
print(tr)
