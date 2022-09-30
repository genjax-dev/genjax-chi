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


@dataclass
class _Logistic(Distribution):
    def random_weighted(self, key, **kwargs):
        key, sub_key = jax.random.split(key)
        v = jax.random.logistic(sub_key, **kwargs)
        _, (w, _) = self.estimate_logpdf(sub_key, **kwargs)
        return key, (w, v)

    def estimate_logpdf(self, key, v, **kwargs):
        w = jnp.sum(jax.scipy.stats.logistic.logpdf(v))
        return key, (w, v)


Logistic = _Logistic()
