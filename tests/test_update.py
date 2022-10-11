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


key = jax.random.PRNGKey(314159)


@genjax.gen
def simple_normal(key):
    key, y1 = genjax.trace("y1", genjax.Normal)(key, (0.0, 1.0))
    key, y2 = genjax.trace("y2", genjax.Normal)(key, (0.0, 1.0))
    return key, y1 + y2


class TestUpdateSimpleNormal:
    def test_simple_normal_update(self, benchmark):
        new_key, tr = jax.jit(genjax.simulate(simple_normal))(key, ())
        jitted = jax.jit(genjax.update(simple_normal))

        new = genjax.ChoiceMap.new({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        new_key, (w, updated, discard) = benchmark(
            jitted, new_key, tr, new, ()
        )
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        _, (score1, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y1"), (0.0, 1.0)
        )
        _, (score2, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

        new = genjax.ChoiceMap.new({("y1",): 2.0, ("y2",): 3.0})
        original_score = tr.get_score()
        new_key, (w, updated, discard) = jitted(new_key, tr, new, ())
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        _, (score1, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y1"), (0.0, 1.0)
        )
        _, (score2, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)


@genjax.gen
def simple_linked_normal(key):
    key, y1 = genjax.trace("y1", genjax.Normal)(key, (0.0, 1.0))
    key, y2 = genjax.trace("y2", genjax.Normal)(key, (y1, 1.0))
    return key, y1 + y2


class TestUpdateSimpleLinkedNormal:
    def test_simple_linked_normal_update(self, benchmark):
        new_key, tr = jax.jit(genjax.simulate(simple_linked_normal))(key, ())
        jitted = jax.jit(genjax.update(simple_linked_normal))

        new = genjax.ChoiceMap.new({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        new_key, (w, updated, discard) = benchmark(
            jitted, new_key, tr, new, ()
        )
        updated_chm = updated.get_choices().strip_metadata()
        y1 = updated_chm["y1"]
        y2 = updated_chm["y2"]
        score1 = genjax.Normal.logpdf(y1, 0.0, 1.0)
        score2 = genjax.Normal.logpdf(y2, y1, 1.0)
        test_score = score1 + score2
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)
