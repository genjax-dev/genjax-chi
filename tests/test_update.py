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
import genjax as gex
import pytest

key = jax.random.PRNGKey(314159)


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


class TestUpdate:
    def test_simple_normal_update(self, benchmark):
        new_key, tr = jax.jit(gex.simulate(simple_normal))(key)
        jitted = jax.jit(gex.update(simple_normal))

        new = gex.ChoiceMap({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        new_key, (w, updated, discard) = benchmark(jitted, new_key, tr, new)
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        test_score = gex.Normal().score(y1) + gex.Normal().score(y2)
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

        new = gex.ChoiceMap({("y1",): 2.0, ("y2",): 3.0})
        original_score = tr.get_score()
        new_key, (w, updated, discard) = jitted(new_key, tr, new)
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        test_score = gex.Normal().score(y1) + gex.Normal().score(y2)
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)
