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

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from genjax._src.core.interpreters.staging import Flag


@genjax.mask
@genjax.gen
def model(x):
    z = genjax.normal(x, 1.0) @ "z"
    return z


class TestMaskCombinator:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(314159)

    def test_mask_simple_normal_true(self, key):
        tr = jax.jit(model.simulate)(key, (True, -4.0))
        assert tr.get_score() == tr.inner.get_score()
        assert tr.get_retval() == genjax.Mask(
            Flag(jnp.array(True)), tr.inner.get_retval()
        )

        tr = jax.jit(model.simulate)(key, (False, -4.0))
        assert tr.get_score() == 0.0
        assert tr.get_retval() == genjax.Mask(
            Flag(jnp.array(False)), tr.inner.get_retval()
        )

    def test_mask_simple_normal_false(self, key):
        tr = jax.jit(model.simulate)(key, (False, 2.0))
        assert tr.get_score() == 0.0
        assert not tr.get_retval().flag

        score, retval = jax.jit(model.assess)(tr.get_sample(), tr.get_args())
        assert score == 0.0
        assert not retval.flag

        _, w = jax.jit(model.importance)(key, C["z"].set(-2.0), tr.get_args())
        assert w == 0.0

    def test_mask_update_weight_to_argdiffs_from_true(self, key):
        # pre-update mask arg is True
        tr = jax.jit(model.simulate)(key, (True, 2.0))
        # mask check arg transition: True --> True
        argdiffs = U.g(
            (Diff.unknown_change(Flag(True)), Diff.no_change(tr.get_args()[1])), C.n()
        )
        w = tr.update(key, argdiffs)[1]
        assert w == tr.inner.update(key, C.n())[1]
        assert w == 0.0
        # mask check arg transition: True --> False
        argdiffs = U.g(
            (Diff.unknown_change(Flag(False)), Diff.no_change(tr.get_args()[1])), C.n()
        )
        w = tr.update(key, argdiffs)[1]
        assert w == -tr.get_score()

    @pytest.mark.skip(reason="This test is currently skipped")
    def test_mask_update_weight_to_argdiffs_from_false(self, key):
        # pre-update mask arg is False
        tr = jax.jit(model.simulate)(key, (False, 2.0))
        # mask check arg transition: False --> True
        argdiffs = U.g(
            (Diff.unknown_change(Flag(True)), Diff.no_change(tr.get_args()[1])), C.n()
        )
        w = tr.update(key, argdiffs)[1]
        assert w == tr.inner.update(key, C.n())[1] + tr.inner.get_score()
        assert w == tr.inner.update(key, C.n())[0].get_score()
        # mask check arg transition: False --> False
        argdiffs = U.g(
            (Diff.unknown_change(Flag(False)), Diff.no_change(tr.get_args()[1])), C.n()
        )
        w = tr.update(key, argdiffs)[1]
        assert w == 0.0
        assert w == tr.get_score()
