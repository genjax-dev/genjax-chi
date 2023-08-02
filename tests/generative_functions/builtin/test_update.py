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

from dataclasses import dataclass

import jax
import pytest

import genjax
from genjax._src.core.typing import FloatArray


class TestUpdateSimpleNormal:
    def test_simple_normal_update(self):
        @genjax.gen
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

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
        _, (score1, _) = genjax.normal.importance(
            key, updated_chm.get_subtree("y1").get_choices(), (0.0, 1.0)
        )
        _, (score2, _) = genjax.normal.importance(
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
        _, (score1, _) = genjax.normal.importance(
            key, updated_chm.get_subtree("y1").get_choices(), (0.0, 1.0)
        )
        _, (score2, _) = genjax.normal.importance(
            key, updated_chm.get_subtree("y2").get_choices(), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_linked_normal_update(self):
        @genjax.gen
        def simple_linked_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(y1, 1.0)
            y3 = genjax.trace("y3", genjax.normal)(y1 + y2, 1.0)
            return y1 + y2 + y3

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
        score1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score2 = genjax.normal.logpdf(y2, y1, 1.0)
        score3 = genjax.normal.logpdf(y3, y1 + y2, 1.0)
        test_score = score1 + score2 + score3
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_hierarchical_normal(self):
        @genjax.gen
        def _inner(x):
            y1 = genjax.trace("y1", genjax.normal)(x, 1.0)
            return y1

        @genjax.gen
        def simple_hierarchical_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", _inner)(y1)
            y3 = genjax.trace("y3", _inner)(y1 + y2)
            return y1 + y2 + y3

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
        score1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score2 = genjax.normal.logpdf(y2, y1, 1.0)
        score3 = genjax.normal.logpdf(y3, y1 + y2, 1.0)
        test_score = score1 + score2 + score3
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_update_weight_correctness(self):
        @genjax.gen
        def simple_linked_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(y1, 1.0)
            y3 = genjax.trace("y3", genjax.normal)(y1 + y2, 1.0)
            return y1 + y2 + y3

        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(simple_linked_normal))(key, ())
        jitted = jax.jit(genjax.update(simple_linked_normal))

        old_y1 = tr["y1"]
        old_y2 = tr["y2"]
        old_y3 = tr["y3"]
        new_y1 = 2.0
        new = genjax.choice_map({("y1",): new_y1})
        key, (_, w, updated, _) = jitted(key, tr, new, ())

        # Test new scores.
        assert updated["y1"] == new_y1
        sel = genjax.select("y1")
        assert updated.project(sel) == genjax.normal.logpdf(new_y1, 0.0, 1.0)
        assert updated["y2"] == old_y2
        sel = genjax.select("y2")
        assert updated.project(sel) == pytest.approx(
            genjax.normal.logpdf(old_y2, new_y1, 1.0), 0.0001
        )
        assert updated["y3"] == old_y3
        sel = genjax.select("y3")
        assert updated.project(sel) == pytest.approx(
            genjax.normal.logpdf(old_y3, new_y1 + old_y2, 1.0), 0.0001
        )

        # Test weight correctness.
        δ_y3 = genjax.normal.logpdf(
            old_y3, new_y1 + old_y2, 1.0
        ) - genjax.normal.logpdf(old_y3, old_y1 + old_y2, 1.0)
        δ_y2 = genjax.normal.logpdf(old_y2, new_y1, 1.0) - genjax.normal.logpdf(
            old_y2, old_y1, 1.0
        )
        δ_y1 = genjax.normal.logpdf(new_y1, 0.0, 1.0) - genjax.normal.logpdf(
            old_y1, 0.0, 1.0
        )
        assert w == pytest.approx((δ_y3 + δ_y2 + δ_y1), 0.0001)

        # Test composition of update calls.
        new_y3 = 2.0
        new = genjax.choice_map({("y3",): new_y3})
        key, (_, w, updated, _) = jitted(key, updated, new, ())
        assert updated["y3"] == 2.0
        correct_w = genjax.normal.logpdf(
            new_y3, new_y1 + old_y2, 1.0
        ) - genjax.normal.logpdf(old_y3, new_y1 + old_y2, 1.0)
        assert w == pytest.approx(correct_w, 0.0001)

    def test_update_pytree_argument(self):
        @dataclass
        class SomePytree(genjax.Pytree):
            x: FloatArray
            y: FloatArray

            def flatten(self):
                return (self.x, self.y), ()

        @genjax.gen
        def simple_linked_normal_with_tree_argument(tree):
            y1 = genjax.trace("y1", genjax.normal)(tree.x, tree.y)
            return y1

        key = jax.random.PRNGKey(314159)
        init_tree = SomePytree(0.0, 1.0)
        key, tr = jax.jit(genjax.simulate(simple_linked_normal_with_tree_argument))(
            key, (init_tree,)
        )
        jitted = jax.jit(genjax.update(simple_linked_normal_with_tree_argument))
        new_y1 = 2.0
        constraints = genjax.choice_map({("y1",): new_y1})
        key, (_, w, updated, _) = jitted(
            key,
            tr,
            constraints,
            (genjax.tree_diff(init_tree, genjax.NoChange),),
        )
        assert updated["y1"] == new_y1
        new_tree = SomePytree(1.0, 2.0)
        key, (_, w, updated, _) = jitted(
            key,
            tr,
            constraints,
            (genjax.tree_diff(new_tree, genjax.UnknownChange),),
        )
        assert updated["y1"] == new_y1
