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
import jax.numpy as jnp
import pytest

import genjax


@genjax.gen
def model(μ):
    x = genjax.normal(μ, 1.0) @ "x"
    return (x - μ) ** 2


class TestVarianceNormal:
    def test_transform(self):
        adev_prog = genjax.adev.lang(model)
        assert isinstance(adev_prog, genjax.adev.ADEVProgram)

    def test_expected_grad(self):
        key = jax.random.PRNGKey(314159)
        sub_keys = jax.random.split(key, 1000)
        adev_prog = genjax.adev.lang(model)
        v, tangents = jax.vmap(adev_prog.grad_estimate, in_axes=(0, None, None))(
            sub_keys, (3.0,), (1.0,)
        )
        assert jnp.mean(tangents) == pytest.approx(0.0, 0.01)
        assert jnp.mean(v) == pytest.approx(1.0, 0.05)


@genjax.gen
def flip_cond(p):
    b = genjax.Bernoulli(p) @ "b"
    return jax.lax.cond(b, lambda _: 0.0, lambda p: p / 2.0, p)


class TestFlipCond:
    def test_transform(self):
        adev_prog = genjax.adev.lang(flip_cond)
        assert isinstance(adev_prog, genjax.adev.ADEVProgram)
