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
from math import pi

import jax
import jax.numpy as jnp

from genjax._src.generative_functions.distributions.distribution import (
    ExactDensity,
)


@dataclass
class _Normal(ExactDensity):
    def sample(self, key, mu, std, **kwargs):
        return mu + std * jax.random.normal(key, **kwargs)

    def logpdf(self, v, mu, std, **kwargs):
        z = (v - mu) / std
        return jnp.sum(
            -1.0
            * (jnp.square(jnp.abs(z)) + jnp.log(2.0 * pi))
            / (2 - jnp.log(std))
        )


Normal = _Normal()
