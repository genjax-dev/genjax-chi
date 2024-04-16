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

import jax  # noqa: I001
import pytest

from genjax.adev import expectation
from genjax.adev import flip_enum
from genjax.adev import expectation
from genjax.adev import add_cost
from genjax.adev import baseline
from genjax.adev import flip_enum
from genjax.adev import flip_reinforce


class TestADEVFlipCond:
    def test_flip_cond_exact_forward_mode_correctness(self):
        @expectation
        def flip_exact_loss(p):
            b = flip_enum(p)
            return jax.lax.cond(
                b,
                lambda _: 0.0,
                lambda p: -p / 2.0,
                p,
            )

        key = jax.random.PRNGKey(314159)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            _, p_tangent = jax.jit(flip_exact_loss.jvp_estimate)(key, (p,), (1.0,))
            assert p_tangent == pytest.approx(p - 0.5, rel=0.0001)

    def test_flip_cond_exact_reverse_mode_correctness(self):
        @expectation
        def flip_exact_loss(p):
            b = flip_enum(p)
            return jax.lax.cond(
                b,
                lambda _: 0.0,
                lambda p: -p / 2.0,
                p,
            )

        key = jax.random.PRNGKey(314159)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            (p_grad,) = jax.jit(flip_exact_loss.grad_estimate)(key, (p,))
            assert p_grad == pytest.approx(p - 0.5, rel=0.0001)

    def test_flip_cond_smoke_test_symbolic_zeros(self):
        @expectation
        def flip_exact_loss(p):
            b = flip_enum(0.3)
            return jax.lax.cond(
                b,
                lambda _: 0.0,
                lambda p: -p / 2.0,
                p,
            )

        key = jax.random.PRNGKey(314159)
        _, p_tangent = jax.jit(flip_exact_loss.jvp_estimate)(key, (0.1,), (1.0,))

    def test_add_cost(self):
        @expectation
        def flip_exact_loss(p):
            add_cost(p**2)
            return 0.0

        key = jax.random.PRNGKey(314159)
        _, p_tangent = jax.jit(flip_exact_loss.jvp_estimate)(key, (0.1,), (1.0,))


class TestBaselineFlip:
    def test_baseline_flip(self):
        @expectation
        def flip_reinforce_loss_no_baseline(p):
            b = flip_reinforce(p)
            v = jax.lax.cond(b, lambda: -1.0, lambda: 1.0)
            return v

        @expectation
        def flip_reinforce_loss(p):
            b = baseline(flip_reinforce)(10.0, p)
            v = jax.lax.cond(b, lambda: -1.0, lambda: 1.0)
            return v + 10.0

        key = jax.random.PRNGKey(314159)
        _, p_tangent_no_baseline = jax.jit(
            flip_reinforce_loss_no_baseline.jvp_estimate
        )(key, (0.1,), (1.0,))

        _, p_tangent = jax.jit(flip_reinforce_loss.jvp_estimate)(key, (0.1,), (1.0,))

        assert p_tangent == pytest.approx(p_tangent_no_baseline, 1e-3)
