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
import jax
from jax._src import abstract_arrays
from dataclasses import dataclass
from genjax.distributions.distribution import Distribution


@dataclass
class _Poisson(Distribution):
    @classmethod
    def abstract_eval(cls, key, lam, shape=None):
        return (
            key,
            abstract_arrays.ShapedArray(shape=shape, dtype=jnp.int),
        )

    def sample(self, key, lam, **kwargs):
        return jax.random.poisson(key, lam, **kwargs)

    def logpdf(self, v, lam, **kwargs):
        return jnp.sum(jax.scipy.stats.poisson.logpmf(v, lam))


Poisson = _Poisson()
