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

import jax
import jax.numpy as jnp

from genjax._src.core.tracetypes import PositiveReals
from genjax._src.generative_functions.distributions.distribution import ExactDensity


@dataclass
class _Cauchy(ExactDensity):
    def sample(self, key, **kwargs):
        return jax.random.cauchy(key, **kwargs)

    def logpdf(self, v, **kwargs):
        return jnp.sum(jax.scipy.stats.cauchy.logpdf(v))

    def get_trace_type(self, key, **kwargs):
        shape = kwargs.get("shape", ())
        return PositiveReals(shape)


Cauchy = _Cauchy()
