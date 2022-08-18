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

# A short example with some higher-order abilities.


def g(key, x):
    key, m1 = genjax.trace("m0", genjax.Bernoulli)(key, x)
    return (key, m1)


def f(key, x):
    def _inner(key):
        key, v = g(key, x)
        key, v2 = genjax.trace("m1", genjax.Bernoulli)(key, x)
        return key, v + v2

    return key, _inner


def toplevel(key, x):
    key, fn = f(key, x)
    key, q = genjax.trace("higher-order", fn)(key)
    return key, q


# Initialize a PRNG.
key = jax.random.PRNGKey(314159)

key, tr = jax.jit(genjax.simulate(toplevel))(key, (0.3,))
print(tr.get_retval())

key, tr = jax.jit(genjax.simulate(toplevel))(key, (0.8,))
print(tr.get_retval())
