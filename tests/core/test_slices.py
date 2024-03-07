# Copyright 2024 MIT Probabilistic Computing Project
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

import genjax
import jax
import jax.numpy as jnp


class TestSlices:
    K = jax.random.PRNGKey(314159)

    @genjax.unfold_combinator(max_length=10)
    @genjax.interpreted_gen_fn
    def random_walk(prev):
        dx = genjax.normal(0.0, 0.1) @ "dx"
        return prev + dx

    def test_slice_unfold(self):
        out = TestSlices.random_walk.simulate(TestSlices.K, (10, 5.0))
        assert out.project[:0] == 0.0
        assert out.project[:0, "dx"] == 0.0
        assert out.project[:, "dx"] == out.get_score()
        f = out.get_choices().filter[:2]
        assert (f.indices == jnp.array([0,1])).all()
        g = out.get_choices().filter[3::2]
        assert (g.indices == jnp.array([3, 5, 7, 9])).all()
