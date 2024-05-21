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
from genjax import ChoiceMapBuilder as C


class TestMaskCombinator:
    def test_mask_simple_normal_true(self):
        @genjax.mask_combinator
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        tr = jax.jit(model.simulate)(key, (True, -4.0))
        assert tr.get_score() == tr.inner.get_score()
        assert tr.get_retval() == genjax.Mask(jnp.array(True), tr.inner.get_retval())
        tr = jax.jit(model.simulate)(key, (False, -4.0))
        assert tr.get_score() == 0.0
        assert tr.get_retval() == genjax.Mask(jnp.array(False), tr.inner.get_retval())

    def test_mask_simple_normal_false(self):
        @genjax.mask_combinator
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        tr = jax.jit(model.simulate)(key, (False, 2.0))
        assert tr.get_score() == 0.0
        assert not tr.get_retval().flag

        score, retval = jax.jit(model.assess)(tr.get_sample(), tr.get_args())
        assert score == 0.0
        assert not retval.flag

        _, w = jax.jit(model.importance)(key, C["z"].set(-2.0), tr.get_args())
        assert w == 0.0
