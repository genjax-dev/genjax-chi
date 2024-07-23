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
from genjax import ChoiceMapBuilder as C
from genjax._src.inference.smc import (
    ImportanceK,
    MultinomialResampling,
    ResidualResampling,
    StratifiedResampling,
    SystematicResampling,
    Target,
)


# TODO: need to look at Julia implementation to see intended composition of resampling strategy with other ones.
class TestResampling:
    key = jax.random.PRNGKey(314159)

    @genjax.gen
    def model():
        _ = genjax.flip(0.5) @ "x"
        _ = genjax.flip(0.7) @ "y"

    target = Target(model, (), C["y"].set(True))
    alg = ImportanceK(target, k_particles=100)
    particles = alg.run_smc(key)

    def test_multinomial_resampling(self):
        multinomial_resample = MultinomialResampling(self.alg, 20, ess_threshold=1.0)
        multinomial_resample.run_smc(self.key)

    def test_stratified_resampling(self):
        stratified_resample = StratifiedResampling(self.alg, 20, ess_threshold=1.0)
        stratified_resample.run_smc(self.key)

    def test_systematic_resampling(self):
        systematic_resample = SystematicResampling(self.alg, 20, ess_threshold=1.0)
        systematic_resample.run_smc(self.key)

    def test_residual_resampling(self):
        residual_resample = ResidualResampling(self.alg, 20, ess_threshold=1.0)
        residual_resample.run_smc(self.key)

    # TODO: not exactly sure what assert to put in all these examples.
