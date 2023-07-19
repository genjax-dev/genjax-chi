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


class TestImportance:
    def test_importance_simple_normal(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        fn = genjax.importance(simple_normal)
        chm = genjax.choice_map({("y1",): 0.5, ("y2",): 0.5})
        key, (_, tr) = fn(key, chm, ())
        out = tr.get_choices()
        y1 = chm[("y1",)]
        y2 = chm[("y2",)]
        _, (score_1, _) = genjax.normal.importance(
            key, chm.get_subtree("y1"), (0.0, 1.0)
        )
        _, (score_2, _) = genjax.normal.importance(
            key, chm.get_subtree("y2"), (0.0, 1.0)
        )
        test_score = score_1 + score_2
        assert y1 == out[("y1",)]
        assert y2 == out[("y2",)]
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_importance_weight_correctness(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

        # Full constraints.
        key = jax.random.PRNGKey(314159)
        chm = genjax.choice_map({("y1",): 0.5, ("y2",): 0.5})
        key, (w, tr) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"]
        y2 = tr["y2"]
        assert y1 == 0.5
        assert y2 == 0.5
        _, (score_1, _) = genjax.normal.importance(
            key, chm.get_subtree("y1"), (0.0, 1.0)
        )
        _, (score_2, _) = genjax.normal.importance(
            key, chm.get_subtree("y2"), (0.0, 1.0)
        )
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(test_score, 0.0001)

        # Partial constraints.
        chm = genjax.choice_map({("y2",): 0.5})
        key, (w, tr) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"]
        y2 = tr["y2"]
        assert y2 == 0.5
        score_1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score_2 = genjax.normal.logpdf(y2, 0.0, 1.0)
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(score_2, 0.0001)
        
        # No constraints.
        chm = genjax.EmptyChoiceMap()
        key, (w, tr) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"]
        y2 = tr["y2"]
        score_1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score_2 = genjax.normal.logpdf(y2, 0.0, 1.0)
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == 0.0
