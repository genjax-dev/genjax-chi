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
    key, m0 = genjax.trace("m0", genjax.Bernoulli)(key, (x,))
    key, m1 = genjax.trace("m1", genjax.Bernoulli)(key, (x,))
    key, m2 = genjax.trace("m2", genjax.Bernoulli)(key, (x,))
    key, m4 = genjax.trace("m3", genjax.Bernoulli)(key, (x,))
    key, m3 = genjax.trace("m4", genjax.Bernoulli)(key, (x,))
    key, m5 = genjax.trace("m5", genjax.Bernoulli)(key, (x,))
    return (key,)


@genjax.gen
def h2(key, x):
    key, m10 = genjax.trace("m10", genjax.Normal)(key, (0.0, 1.0))
    key, m11 = genjax.trace("m11", genjax.Normal)(key, (0.0, 1.0))
    key, m12 = genjax.trace("m12", genjax.Normal)(key, (0.0, 1.0))
    key, m13 = genjax.trace("m13", genjax.Normal)(key, (0.0, 1.0))
    key, m14 = genjax.trace("m14", genjax.Normal)(key, (0.0, 1.0))
    return (key,)


sw = genjax.SwitchCombinator([h1, h2])


class TestSwitch:
    def test_simple_two_branch_switch_simulate(self, benchmark):
        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(genjax.simulate(sw))
        key, tr = benchmark(jitted, key, (1, 0.3))

    def test_simple_two_branch_switch_importance(self, benchmark):
        key = jax.random.PRNGKey(314159)
        chm = genjax.ChoiceMap.new({("m11",): 1.0})
        jitted = jax.jit(genjax.importance(sw))
        key, (w, tr) = benchmark(jitted, key, chm, (1, 0.6))

    def test_simple_two_branch_switch_update(self, benchmark):
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(sw))(key, (1, 0.3))
        chm = genjax.ChoiceMap.new({("m12",): 2.0})
        jitted = jax.jit(genjax.update(sw))
        key, (w, tr, d) = benchmark(jitted, key, tr, chm, (1, 0.6))
