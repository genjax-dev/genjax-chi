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
import pytest
from genjax import ChoiceMap as C
from genjax import Diff, RemoveSelectionUpdateSpec


class TestScanSimpleNormal:
    def test_scan_simple_normal(self):
        @genjax.scan_combinator(max_length=10)
        @genjax.gen
        def scanner(x, c):
            z = genjax.normal(x, 1.0) @ "z"
            return z, None

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(scanner.simulate)(sub_key, (0.01, None))
        scan_score = tr.get_score()
        sel = genjax.Selection.at[..., "z"]
        assert tr.project(key, RemoveSelectionUpdateSpec(sel)) == scan_score

    def test_scan_simple_normal_importance(self):
        @genjax.scan_combinator(max_length=10)
        @genjax.gen
        def scanner(x, c):
            z = genjax.normal(x, 1.0) @ "z"
            return z, None

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, w, bwd_spec = jax.jit(scanner.importance)(
                sub_key, C.n.at[i, "z"].set(0.5), (0.01, None)
            )
            assert tr.get_sample()[i, "z"] == 0.5
            value = tr.get_sample()[i, "z"]
            prev = tr.get_sample()[i - 1, "z"]
            assert w == genjax.normal.assess(C.v(value), (prev, 1.0))[0]

    def test_scan_simple_normal_update(self):
        @genjax.scan_combinator(max_length=10)
        @genjax.gen
        def scanner(x, c):
            z = genjax.normal(x, 1.0) @ "z"
            return z, None

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, w, bwd_spec = jax.jit(scanner.importance)(
                sub_key, C.n.at[i, "z"].set(0.5), (0.01, None)
            )
            new_tr, w, rd, bwd_spec = jax.jit(scanner.update)(
                sub_key,
                tr,
                C.n.at[i, "z"].set(1.0),
                Diff.no_change((0.01, None)),
            )
            assert new_tr.get_sample()[i, "z"] == 1.0
            assert tr.get_score() + w == pytest.approx(new_tr.get_score(), 0.0001)
