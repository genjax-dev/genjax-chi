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
import pytest

import genjax


class TestSimulate:
    def test_simple_normal_simulate(self):
        @genjax.gen(genjax.Static)
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        fn = jax.jit(simple_normal.simulate)
        key, sub_key = jax.random.split(key)
        tr = fn(sub_key, ())
        chm = tr.get_choices()
        (_, score1) = genjax.normal.importance(key, chm.get_submap("y1"), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, chm.get_submap("y2"), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_multiple_returns(self):
        @genjax.gen(genjax.Static)
        def simple_normal_multiple_returns():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        fn = jax.jit(simple_normal_multiple_returns.simulate)
        tr = fn(sub_key, ())
        y1_ = tr["y1"]
        y2_ = tr["y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (_, score1) = genjax.normal.importance(key, genjax.choice_value(y1), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, genjax.choice_value(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_hierarchical_simple_normal_multiple_returns(self):
        @genjax.gen(genjax.Static)
        def _submodel():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1, y2

        @genjax.gen(genjax.Static)
        def hierarchical_simple_normal_multiple_returns():
            y1, y2 = genjax.trace("y1", _submodel)()
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        fn = jax.jit(hierarchical_simple_normal_multiple_returns.simulate)
        tr = fn(sub_key, ())
        y1_ = tr["y1", "y1"]
        y2_ = tr["y1", "y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (_, score1) = genjax.normal.importance(key, genjax.choice_value(y1), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, genjax.choice_value(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)
