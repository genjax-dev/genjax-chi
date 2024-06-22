# Copyright 2023 MIT Probabilistic Computing Project
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
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import SelectionBuilder as S
from genjax import UpdateProblemBuilder as U


class TestScanSimpleNormal:
    def test_scan_simple_normal(self):
        @genjax.scan(n=10)
        @genjax.gen
        def scanner(x, c):
            z = genjax.normal(x, 1.0) @ "z"
            return z, None

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(scanner.simulate)(sub_key, (0.01, None))
        scan_score = tr.get_score()
        sel = S[..., "z"]
        assert tr.project(key, sel) == scan_score

    def test_scan_of_repeat(self):
        @genjax.scan(n=10)
        @genjax.gen
        def kernel(x, _):
            d1 = genjax.normal(0.0, 1.0) @ "d1"
            d2 = genjax.normal.repeat(n=4)(0.0, 1.0) @ "d2"

            return x + d1 + d2, None

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(kernel.simulate)(sub_key, (jnp.zeros(4), None))
        assert tr.project(S[..., "d1"]) + tr.project(S[..., "d2"]) == tr.get_score()
        assert (
            tr.project(genjax.ChoiceMapBuilder[jnp.array([0, 1, 2, 3, 4]), ...])
            + tr.project(genjax.ChoiceMapBuilder[jnp.array([5, 6, 7, 8, 9]), ...])
            == tr.get_score()
        )

    def test_scan_simple_normal_importance(self):
        @genjax.scan(n=10)
        @genjax.gen
        def scanner(x, c):
            z = genjax.normal(x, 1.0) @ "z"
            return z, None

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, w = jax.jit(scanner.importance)(
                sub_key, C[i, "z"].set(0.5), (0.01, None)
            )
            assert tr.get_sample()[i, "z"].unmask() == 0.5
            value = tr.get_sample()[i, "z"].unmask()
            prev = tr.get_sample()[i - 1, "z"].unmask()
            assert w == genjax.normal.assess(C.v(value), (prev, 1.0))[0]

    def test_scan_simple_normal_update(self):
        @genjax.scan(n=10)
        @genjax.gen
        def scanner(x, c):
            z = genjax.normal(x, 1.0) @ "z"
            return z, None

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, w = jax.jit(scanner.importance)(
                sub_key, C[i, "z"].set(0.5), (0.01, None)
            )
            new_tr, w, _rd, _bwd_problem = jax.jit(scanner.update)(
                sub_key,
                tr,
                U.g(
                    Diff.no_change((0.01, None)),
                    C[i, "z"].set(1.0),
                ),
            )
            assert new_tr.get_sample()[i, "z"].unmask() == 1.0
            assert tr.get_score() + w == pytest.approx(new_tr.get_score(), 0.0001)
