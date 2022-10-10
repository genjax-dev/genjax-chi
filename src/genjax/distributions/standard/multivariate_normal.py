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

from genjax.distributions.distribution import ExactDistribution


@dataclass
class _MultivariateNormal(ExactDistribution):
    def sample(self, key, mean, cov, **kwargs):
        return jax.random.multivariate_normal(key, mean, cov, **kwargs)

    def logpdf(self, v, mean, cov, **kwargs):
        return jnp.sum(
            jax.scipy.stats.multivariate_normal.logpdf(v, mean, cov)
        )


MvNormal = _MultivariateNormal()
