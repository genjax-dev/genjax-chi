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

import dataclasses
from typing import Any

import jax
import pytest

import genjax


@dataclasses.dataclass
class CustomTree(genjax.Pytree):
    x: Any
    y: Any

    def flatten(self):
        return (self.x, self.y), ()


@genjax.gen
def simple_normal(custom_tree):
    y1 = genjax.trace("y1", genjax.normal)(custom_tree.x, 1.0)
    y2 = genjax.trace("y2", genjax.normal)(custom_tree.y, 1.0)
    return CustomTree(y1, y2)


@dataclasses.dataclass
class _CustomNormal(genjax.ExactDensity):
    def logpdf(self, v, custom_tree):
        return genjax.normal.logpdf(v, custom_tree.x, custom_tree.y)

    def sample(self, key, custom_tree):
        return genjax.normal.sample(key, custom_tree.x, custom_tree.y)


CustomNormal = _CustomNormal()


@genjax.gen
def custom_normal(custom_tree):
    y = genjax.trace("y", CustomNormal)(custom_tree)
    return CustomTree(y, y)


class TestCustomPytree:
    def test_simple_normal_simulate(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        fn = jax.jit(genjax.simulate(simple_normal))
        tr = fn(key, (init_tree,))
        chm = tr.get_choices()
        (score1, _) = genjax.normal.importance(
            key, chm.get_subtree("y1").get_choices(), (init_tree.x, 1.0)
        )
        (score2, _) = genjax.normal.importance(
            key, chm.get_subtree("y2").get_choices(), (init_tree.y, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_custom_normal_simulate(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        fn = jax.jit(genjax.simulate(custom_normal))
        tr = fn(key, (init_tree,))
        chm = tr.get_choices()
        (score, _) = genjax.normal.importance(
            key, chm.get_subtree("y").get_choices(), (init_tree.x, init_tree.y)
        )
        test_score = score
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_importance(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        chm = genjax.choice_map({"y1": 5.0})
        fn = jax.jit(genjax.importance(simple_normal))
        (w, tr) = fn(key, chm, (init_tree,))
        chm = tr.get_choices()
        (score1, _) = genjax.normal.importance(
            key, chm.get_subtree("y1").get_choices(), (init_tree.x, 1.0)
        )
        (score2, _) = genjax.normal.importance(
            key, chm.get_subtree("y2").get_choices(), (init_tree.y, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)
        assert w == pytest.approx(score1, 0.01)
