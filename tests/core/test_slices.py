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
import pytest
from genjax._src.core.typing import FloatArray


class TestSlices:
    K = jax.random.PRNGKey(314159)

    @genjax.unfold_combinator(max_length=10)
    @genjax.interpreted_gen_fn
    def random_walk(prev):
        dx = genjax.normal(0.0, 0.1) @ "dx"
        return prev + dx

    @genjax.interpreted_gen_fn
    def contains_random_walk():
        # Draws a starting point and walks from there
        base = genjax.normal(10.0, 0.1) @ "x0"
        return TestSlices.random_walk(10, base) @ "walk"

    def test_slice_unfold(self):
        out = TestSlices.random_walk.simulate(TestSlices.K, (10, 5.0))
        assert out.project[:0] == 0.0
        assert out.project[:0, "dx"] == 0.0
        assert out.project[:, "dx"] == out.get_score()
        choices = out.get_choices()
        f = choices.filter[:2]
        assert (f.indices == jnp.array([0, 1])).all()
        g = choices.filter[3::2]
        assert (g.indices == jnp.array([3, 5, 7, 9])).all()
        assert (choices.filter[-3:].indices == jnp.array([7, 8, 9])).all()
        assert (choices.filter[:-6].indices == jnp.array([0, 1, 2, 3])).all()

    def test_slice_wrapped_unfold(self):
        out = TestSlices.contains_random_walk.simulate(TestSlices.K, ())
        steps: FloatArray = out["walk", :, "dx"]
        assert out.get_retval() == pytest.approx(out["x0"] + steps.cumsum())
        choices = out.get_choices()
        assert choices.filter["x0"]["x0"] == jnp.array(9.989177)
        assert choices.filter["x0"]["walk"] == genjax.EmptyChoice()
        assert choices.filter["walk"]["x0"] == genjax.EmptyChoice()
        assert (
            choices.filter["walk"]["walk", :].has_submap("dx") == jnp.ones((10,))
        ).all()
