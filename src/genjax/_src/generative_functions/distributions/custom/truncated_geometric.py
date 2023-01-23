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

from dataclasses import dataclass

import jax.numpy as jnp

from genjax._src.generative_functions.distributions.distribution import ExactDensity
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    TFPGeometric,
)


@dataclass
class TruncatedGeometric(ExactDensity):
    def sample(key, p, m):
        pass

    def _log_geometric_remainder(self, p, m):
        return jnp.log(p) + m * jnp.log(1 - p) - jnp.log(p)

    def logpdf(self, v, p, m):
        pr = TFPGeometric.logpdf(v, p)
        remainder = self._log_geometric_remainder(p, m)
        return pr + remainder
