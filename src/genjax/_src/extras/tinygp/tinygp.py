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

from typing import Any

import tinygp

from genjax._src.generative_functions.distributions.distribution import ExactDensity


class GaussianProcess(ExactDensity):
    kernel: Any

    def sample(self, key, X, **kwargs):
        gp = tinygp.GaussianProcess(self.kernel, X)
        y = gp.sample(key, **kwargs)
        return y

    def logpdf(self, y, X, **kwargs):
        gp = tinygp.GaussianProcess(self.kernel, X)
        return gp.log_probability(y)

    def condition(self, y, X):
        gp = tinygp.GaussianProcess(self.kernel, X)
        return ConditionedGaussianProcess(gp.condition(y))


class ConditionedGaussianProcess(ExactDensity):
    conditioned: Any

    def flatten(self):
        return (self.conditioned,), ()

    def sample(self, key, X, **kwargs):
        y = self.conditioned.sample(key, **kwargs)
        return y

    def logpdf(self, y, X, **kwargs):
        return self.conditioned.log_probability(y)
