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
from jax._src import abstract_arrays
from genjax.distributions.trace import DistributionTrace
from genjax.primitive import PrimitiveGenerativeFunction


class _Bernoulli(PrimitiveGenerativeFunction):
    def abstract_eval(self, key, p, shape=()):
        return (
            key,
            abstract_arrays.ShapedArray(shape=shape, dtype=bool),
        )

    def simulate(self, key, args, **kwargs):
        key, sub_key = jax.random.split(key)
        v = jax.random.bernoulli(sub_key, *args, **kwargs)
        tr = DistributionTrace(
            Bernoulli, v, jnp.sum(jax.scipy.stats.bernoulli.logpmf(v, *args))
        )
        return (key, tr)

    def importance(self, key, chm, args, **kwargs):
        v = chm.get_value()
        weight = jnp.sum(jax.scipy.stats.bernoulli.logpmf(v, *args))
        return key, (weight, DistributionTrace(Bernoulli, v, weight))

    def flatten(self):
        return (), ()

    def unflatten(self, values, slices):
        return _Bernoulli()


Bernoulli = _Bernoulli()
