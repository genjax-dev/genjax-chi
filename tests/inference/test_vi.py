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
import pytest

import genjax
from genjax import ChoiceMapBuilder as C


class TestVI:
    def test_normal_normal_tight_variance(self):
        @genjax.gen
        def model(v):
            mu = genjax.normal(0.0, 10.0) @ "mu"
            _ = genjax.normal(mu, 0.1) @ "v"

        @genjax.marginal()
        @genjax.gen
        def guide(target):
            (v,) = target.args
            _ = genjax.vi.normal_reparam(v, 0.1) @ "mu"

        key = jax.random.PRNGKey(314159)
        elbo_grad = genjax.vi.ELBO(
            guide, lambda v: genjax.Target(model, (v,), C["v"].set(3.0))
        )
        v = 0.1
        jitted = jax.jit(elbo_grad)
        for _ in range(200):
            (v_grad,) = jitted(key, (v,))
            v -= 1e-3 * v_grad
        assert v == pytest.approx(3.0, 5e-2)
