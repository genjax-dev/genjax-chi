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
def simple_normal(key):
    key, y1 = genjax.trace("y1", genjax.Normal)(key, 0.0, 1.0)
    key, y2 = genjax.trace("y2", genjax.Normal)(key, 0.0, 1.0)
    return (key,)


@genjax.gen
def simple_bernoulli(key):
    key, y1 = genjax.trace("y1", genjax.Bernoulli)(key, 0.3)
    return (key,)


switch = genjax.SwitchCombinator([simple_normal, simple_bernoulli])


class TestSimulate:
    def test_switch_simulate(self):
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(switch.simulate)(key, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        assert score == genjax.Normal.logpdf(
            v1, 0.0, 1.0
        ) + genjax.Normal.logpdf(v2, 0.0, 1.0)
        key, tr = jax.jit(switch.simulate)(key, (1,))
        flip = tr["y1"]
        score = tr.get_score()
        assert score == genjax.Bernoulli.logpdf(flip, 0.3)

    def test_switch_importance(self):
        key = jax.random.PRNGKey(314159)
        chm = genjax.EmptyChoiceMap()
        key, (w, tr) = jax.jit(switch.importance)(key, chm, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        assert score == genjax.Normal.logpdf(
            v1, 0.0, 1.0
        ) + genjax.Normal.logpdf(v2, 0.0, 1.0)
        assert w == 0.0
        key, (w, tr) = jax.jit(switch.importance)(key, chm, (1,))
        flip = tr["y1"]
        score = tr.get_score()
        assert score == genjax.Bernoulli.logpdf(flip, 0.3)
        assert w == 0.0
        chm = genjax.BuiltinChoiceMap.new({"y1": True})
        key, (w, tr) = jax.jit(switch.importance)(key, chm, (1,))
        flip = tr["y1"]
        score = tr.get_score()
        assert score == genjax.Bernoulli.logpdf(flip, 0.3)
        assert w == score
