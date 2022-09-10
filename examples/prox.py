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
def h(key, mu, std):
    key, m1 = genjax.trace("m1", genjax.Normal)(key, (mu, std))
    key, m2 = genjax.trace("m2", genjax.Normal)(key, (mu, std))
    return (key, m1 + m2)


key = jax.random.PRNGKey(314159)
marginalized = genjax.ChoiceMapDistribution(h, genjax.Selection(["m1"]), None)

trace_type = genjax.get_trace_type(marginalized)(key, (5.0, 1.0))
print(trace_type)
key, tr = jax.jit(genjax.simulate(marginalized))(key, (5.0, 1.0))
print(tr)
