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

import jax.numpy as jnp

import genjax


class TestCategorical:
    def test_scipy_vs_tf(self):
        logits = jnp.array([0.5, 0.5])
        assert genjax.TFPCategorical.logpdf(0, logits) == genjax.Categorical.logpdf(
            0, logits
        )
        assert genjax.TFPCategorical.logpdf(1, logits) == genjax.Categorical.logpdf(
            1, logits
        )
