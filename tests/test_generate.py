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


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


class TestGenerate:
    def test_simple_normal_generate(self, benchmark):
        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(gex.importance(simple_normal))
        chm = {("y1",): 0.5, ("y2",): 0.5}
        w, tr = benchmark(jitted, chm, key)
        out = tr.get_choices()
        y1 = chm[("y1",)]
        y2 = chm[("y2",)]
        test_score = gex.Normal().score(y1) + gex.Normal().score(y2)
        assert chm == out
        assert tr.get_score() == pytest.approx(test_score, 0.01)
