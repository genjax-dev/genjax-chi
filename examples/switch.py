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
def model1(key):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    key, q = genjax.trace("y", genjax.Bernoulli)(key, (0.3,))
    return (key,)


@genjax.gen
def model2(key):
    key, x = genjax.trace("y", genjax.Bernoulli)(key, (0.3,))
    return (key,)


@genjax.gen
def model3(key):
    key, x = genjax.trace("z", genjax.Uniform)(key, (0.5, 3.0))
    key, y = genjax.trace("m", genjax.Normal)(key, (0.3, 2.0))
    return (key,)


sw = genjax.Switch([model1, model2, model3])

key = jax.random.PRNGKey(314159)
trace_type = genjax.get_trace_type(sw)(key, (1,))
key, tr = genjax.simulate(sw)(key, (1,))
print(trace_type)
chm = genjax.ChoiceMap.new({("z",): 2.0})
key, (w, new, d) = jax.jit(genjax.update(sw))(key, tr, chm, (2,))
d.dump()
