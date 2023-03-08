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


@genjax.gen
def simple_normal():
    y1 = genjax.normal(0.0, 1.0) @ "y1"
    y2 = genjax.normal(0.0, 1.0) @ "y2"
    return y1 + y2


class TestAssessSimpleNormal:
    def test_simple_normal_assess(self):
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(simple_normal))(key, ())
        jitted = jax.jit(genjax.assess(simple_normal))
        chm = tr.get_choices().strip()
        key, (retval, score) = jitted(key, chm, ())
        assert score == tr.get_score()
