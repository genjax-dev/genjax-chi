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


class TestSimulate:
    def test_simple_normal_simulate(self):
        fn = genjax.simulate(simple_normal)
        new_key, tr = fn(key, ())
        chm = tr.get_choices()
        _, (score1, _) = genjax.Normal.importance(
            key, chm.get_subtree("y1").get_choices(), (0.0, 1.0)
        )
        _, (score2, _) = genjax.Normal.importance(
            key, chm.get_subtree("y2").get_choices(), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)
