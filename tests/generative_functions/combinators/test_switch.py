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
from genjax import diff, NoChange, UnknownChange


@genjax.gen
def simple_normal():
    y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
    y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)


@genjax.gen
def simple_bernoulli():
    y3 = genjax.trace("y3", genjax.bernoulli)(0.3)


switch = genjax.SwitchCombinator([simple_normal, simple_bernoulli])


class TestSimulate:
    def test_switch_simulate(self):
        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        key, tr = jitted(key, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        assert score == genjax.normal.logpdf(v1, 0.0, 1.0) + genjax.normal.logpdf(
            v2, 0.0, 1.0
        )
        assert tr.get_args() == (0,)
        key, tr = jitted(key, (1,))
        flip = tr["y3"]
        score = tr.get_score()
        assert score == genjax.bernoulli.logpdf(flip, 0.3)
        assert tr.get_args() == (1,)

    def test_switch_importance(self):
        key = jax.random.PRNGKey(314159)
        chm = genjax.EmptyChoiceMap()
        jitted = jax.jit(switch.importance)
        key, (w, tr) = jitted(key, chm, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        assert score == genjax.normal.logpdf(v1, 0.0, 1.0) + genjax.normal.logpdf(
            v2, 0.0, 1.0
        )
        assert w == 0.0
        key, (w, tr) = jitted(key, chm, (1,))
        flip = tr["y3"]
        score = tr.get_score()
        assert score == genjax.bernoulli.logpdf(flip, 0.3)
        assert w == 0.0
        chm = genjax.choice_map({"y3": True})
        key, (w, tr) = jitted(key, chm, (1,))
        flip = tr["y3"]
        score = tr.get_score()
        assert score == genjax.bernoulli.logpdf(flip, 0.3)
        assert w == score

    def test_switch_update_single_branch_no_change(self):
        key = jax.random.PRNGKey(314159)
        switch = genjax.SwitchCombinator([simple_normal])
        key, tr = jax.jit(switch.simulate)(key, (0,))
        v1 = tr["y1"]
        v2 = tr["y2"]
        score = tr.get_score()
        key, (_, _, tr, _) = jax.jit(switch.update)(
            key, tr, genjax.empty_choice_map(), (diff(0, NoChange),))
        assert score == tr.get_score()
        assert v1 == tr["y1"]
        assert v2 == tr["y2"]

    def test_switch_update_updates_score(self):
        key = jax.random.PRNGKey(314159)

        regular_stddev = 1.
        outlier_stddev = 10.
        sample_value = 2.

        @genjax.gen
        def regular():
            x = genjax.normal(0., regular_stddev) @ 'x'
            return x

        @genjax.gen
        def outlier():
            x = genjax.normal(0., outlier_stddev) @ 'x'
            return x

        switch = genjax.SwitchCombinator([regular, outlier])
        key, importance_key = jax.random.split(key)
            
        _, (wt, tr) = switch.importance(
            importance_key, genjax.choice_map({"x": sample_value}), (0,))
        assert tr.chm.index == 0
        assert tr.get_score() == genjax.normal.logpdf(
            sample_value, 0., regular_stddev)
        assert wt == tr.get_score()

        key, update_key = jax.random.split(key)
        _, (_, new_wt, new_tr, _) = switch.update(
            update_key, tr, genjax.empty_choice_map(), (1,))
        assert new_tr.chm.index == 1
        # These both fail:
        assert new_tr.get_score() != tr.get_score()
        assert new_tr.get_score() == genjax.normal.logpdf(
            sample_value, 0., outlier_stddev)
