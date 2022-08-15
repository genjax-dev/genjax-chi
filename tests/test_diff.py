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
import genjax as gex

key = jax.random.PRNGKey(314159)


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


class TestDiff:
    def test_simple_normal_diff(self, benchmark):
        new_key, tr = jax.jit(gex.simulate(simple_normal))(key)
        original = tr.get_choices()[("y1",)]
        new = gex.ChoiceMap({("y1",): 2.0})
        jitted = jax.jit(gex.diff(simple_normal))
        new_key, (w, ret) = benchmark(jitted, key, tr, new)
