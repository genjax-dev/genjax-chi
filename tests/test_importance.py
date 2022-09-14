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
import pytest

key = jax.random.PRNGKey(314159)


@genjax.gen
def simple_normal(key):
    key, y1 = genjax.trace("y1", genjax.Normal)(key, (0.0, 1.0))
    key, y2 = genjax.trace("y2", genjax.Normal)(key, (0.0, 1.0))
    return key, y1 + y2


class TestImportance:
    def test_simple_normal_importance(self, benchmark):
        jitted = jax.jit(genjax.importance(simple_normal))
        chm = genjax.ChoiceMap.new({("y1",): 0.5, ("y2",): 0.5})
        new_key, (w, tr) = benchmark(jitted, key, chm, ())
        out = tr.get_choices()
        y1 = chm[("y1",)]
        y2 = chm[("y2",)]
        _, (score1, _) = genjax.Normal.importance(
            key, chm.get_choice("y1"), (0.0, 1.0)
        )
        _, (score2, _) = genjax.Normal.importance(
            key, chm.get_choice("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert y1 == out[("y1",)]
        assert y2 == out[("y2",)]
        assert tr.get_score() == pytest.approx(test_score, 0.01)
