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
from genjax.distributions.distribution import Distribution
from math import pi


@dataclass
class _Normal(Distribution):
    def random_weighted(self, key, mu, std, **kwargs):
        key, sub_key = jax.random.split(key)
        v = mu + std * jax.random.normal(sub_key, **kwargs)
        _, (w, _) = self.estimate_logpdf(sub_key, v, mu, std, **kwargs)
        return key, (w, v)

    def estimate_logpdf(self, key, v, mu, std, **kwargs):
        z = (v - mu) / std
        w = jnp.sum(
            -1.0
            * (jnp.square(jnp.abs(z)) + jnp.log(2.0 * pi))
            / (2 - jnp.log(std))
        )
        return key, (w, v)


Normal = _Normal()
