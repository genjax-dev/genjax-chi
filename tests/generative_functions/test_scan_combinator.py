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
import penzai.pz as pz
import pytest
from genjax import ChoiceMapBuilder as C
from genjax import (
    ChoiceMapConstraint,
    ChoiceMapEditRequest,
    IndexEditRequest,
    LambdaEditRequest,
)
from genjax.typing import FloatArray


class TestScanUpdate:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(314159)

    def test_scan_update(self, key):
        @pz.pytree_dataclass
        class A(genjax.Pytree):
            x: FloatArray

        @genjax.gen
        def step(b, a):
            return genjax.normal(b + a.x, 1e-6) @ "b", None

        @genjax.gen
        def model(k):
            return step.scan(n=3)(k, A(jnp.array([1.0, 2.0, 3.0]))) @ "steps"

        k1, k2 = jax.random.split(key)
        tr = model.simulate(k1, (jnp.array(1.0),))
        u, w, _, _ = tr.update(k2, C["steps", 1, "b"].set(99.0))
        assert jnp.allclose(
            u.get_choices()["steps", ..., "b"], jnp.array([2.0, 99.0, 7.0]), atol=0.1
        )
        assert w < -100.0


class TestScanIndexEdit:
    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(314159)

    def test_scan_index_edit(self, key):
        @genjax.gen
        def step(b, a):
            return genjax.normal(b + a, 1e-6) @ "b", None

        model = step.scan(n=3)

        k1, k2 = jax.random.split(key)
        tr = model.simulate(k1, (0.0, jnp.array([0.0, 1.0, 2.0])))
        request = IndexEditRequest(
            jnp.array(1),
            LambdaEditRequest(
                lambda trace_args: ChoiceMapEditRequest(
                    trace_args, ChoiceMapConstraint(C["b"].set(99.0))
                )
            ),
        )
        u, w, _, _ = tr.edit(k2, request)
