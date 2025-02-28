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


import jax
import jax.numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C


class TestMixture:
    def test_mix_basic(self):
        # Define two simple component functions
        @genjax.gen
        def comp1(x):
            return genjax.normal(x, 1.0) @ "y"

        @genjax.gen
        def comp2(x):
            return genjax.normal(x + 2.0, 0.5) @ "y"

        # Create mixture model
        mixture = genjax.mix(comp1, comp2)

        # Test simulation
        key = jax.random.key(0)
        logits = jnp.array([-0.1, -0.2])
        trace = mixture.simulate(key, (logits, (0.0,), (0.0,)))

        # Check structure
        chm = trace.get_choices()
        assert "mixture_component" in chm
        assert ("component_sample", "y") in chm

        # Test assessment
        choices = C["mixture_component"].set(0) | C["component_sample", "y"].set(1.0)
        score, _ = mixture.assess(choices, (logits, (0.0,), (0.0,)))
        assert jnp.isfinite(score)

    def test_mix_then_simulate(self):
        """GEN-1025 brought up an issue where calling `genjax.mix` between two calls to simulate
        would make the second call fail (somehow capturing the internal mix genfn)"""

        @genjax.gen
        def f():
            return genjax.uniform(0.0, 1.1) @ "uniform"

        @genjax.gen
        def g2():
            return f() @ "f"

        key = jax.random.key(0)
        tr = g2.simulate(key, ())
        genjax.mix(genjax.flip, genjax.flip)
        assert tr == g2.simulate(key, ())
