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
from dataclasses import dataclass
from genjax.core.tracetypes import RealInterval
from genjax.distributions.distribution import Distribution


@dataclass
class _Beta(Distribution):
    def sample(self, key, a, b, **kwargs):
        return jax.random.beta(key, a, b, **kwargs)

    def logpdf(self, key, v, a, b, **kwargs):
        return jnp.sum(jax.scipy.stats.beta.logpdf(v, a, b))

    def get_trace_type(self, key, a, b, **kwargs):
        shape = kwargs.get("shape", ())
        return RealInterval(shape, a, b)


Beta = _Beta()
