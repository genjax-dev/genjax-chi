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
def h(key, x):
    key, m1 = genjax.trace("m0", genjax.Bernoulli)(key, (x,))
    return (key, m1)


@genjax.gen
def g(key, x):
    key, m1 = genjax.trace("m0", h)(key, (x,))
    return (key, m1)


@genjax.gen
def f(key, x):
    key, m0 = genjax.trace("m0", genjax.Bernoulli, shape=(3, 3))(key, (x,))
    key, m4 = genjax.trace("m4", g)(key, (x,))
    key, m5 = genjax.trace("m5", genjax.Normal)(key, (0.0, 1.0))
    return key, (2 * m0 * m4, m5)


key = jax.random.PRNGKey(314159)
trace_type = genjax.get_trace_type(f)(key, 0.3)
print(trace_type)
