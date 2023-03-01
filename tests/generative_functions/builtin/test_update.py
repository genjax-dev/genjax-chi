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


@genjax.gen
def simple_normal():
    y1 = genjax.trace("y1", genjax.Normal)(0.0, 1.0)
    y2 = genjax.trace("y2", genjax.Normal)(0.0, 1.0)
    return y1 + y2


class TestUpdateSimpleNormal:
    def test_simple_normal_update(self):
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(simple_normal))(key, ())
        jitted = jax.jit(genjax.update(simple_normal))

        new = genjax.choice_map({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        key, (_, w, updated, discard) = jitted(key, tr, new, ())
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        _, (score1, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y1").get_choices(), (0.0, 1.0)
        )
        _, (score2, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y2").get_choices(), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

        new = genjax.choice_map({("y1",): 2.0, ("y2",): 3.0})
        original_score = tr.get_score()
        key, (_, w, updated, discard) = jitted(key, tr, new, ())
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        _, (score1, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y1").get_choices(), (0.0, 1.0)
        )
        _, (score2, _) = genjax.Normal.importance(
            key, updated_chm.get_subtree("y2").get_choices(), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)


@genjax.gen
def simple_linked_normal():
    y1 = genjax.trace("y1", genjax.Normal)(0.0, 1.0)
    y2 = genjax.trace("y2", genjax.Normal)(y1, 1.0)
    y3 = genjax.trace("y3", genjax.Normal)(y1 + y2, 1.0)
    return y1 + y2 + y3


class TestUpdateSimpleLinkedNormal:
    def test_simple_linked_normal_update(self):
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(simple_linked_normal))(key, ())
        jitted = jax.jit(genjax.update(simple_linked_normal))

        new = genjax.choice_map({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        key, (_, w, updated, discard) = jitted(key, tr, new, ())
        updated_chm = updated.get_choices().strip()
        y1 = updated_chm["y1"]
        y2 = updated_chm["y2"]
        y3 = updated_chm["y3"]
        score1 = genjax.Normal.logpdf(y1, 0.0, 1.0)
        score2 = genjax.Normal.logpdf(y2, y1, 1.0)
        score3 = genjax.Normal.logpdf(y3, y1 + y2, 1.0)
        test_score = score1 + score2 + score3
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)


@genjax.gen
def _inner(x):
    y1 = genjax.trace("y1", genjax.Normal)(x, 1.0)
    return y1


@genjax.gen
def simple_hierarchical_normal():
    y1 = genjax.trace("y1", genjax.Normal)(0.0, 1.0)
    y2 = genjax.trace("y2", _inner)(y1)
    y3 = genjax.trace("y3", _inner)(y1 + y2)
    return y1 + y2 + y3


class TestUpdateSimpleHierarchicalNormal:
    def test_simple_hierarchical_normal(self):
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(simple_hierarchical_normal))(key, ())
        jitted = jax.jit(genjax.update(simple_hierarchical_normal))

        new = genjax.choice_map({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        key, (_, w, updated, discard) = jitted(key, tr, new, ())
        updated_chm = updated.get_choices().strip()
        y1 = updated_chm["y1"]
        y2 = updated_chm["y2", "y1"]
        y3 = updated_chm["y3", "y1"]
        assert y1 == new["y1"]
        assert y2 == original_chm["y2", "y1"]
        assert y3 == original_chm["y3", "y1"]
        score1 = genjax.Normal.logpdf(y1, 0.0, 1.0)
        score2 = genjax.Normal.logpdf(y2, y1, 1.0)
        score3 = genjax.Normal.logpdf(y3, y1 + y2, 1.0)
        test_score = score1 + score2 + score3
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)
