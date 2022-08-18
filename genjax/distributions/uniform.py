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
from genjax.distributions import Distribution


class Uniform(Distribution):
    def abstract_eval(self, key, *params, shape=(), **kwargs):
        return (
            key,
            abstract_arrays.ShapedArray(shape=shape, dtype=jnp.float32),
        )

    def abstract_eval_batched(
        self, key, *params, batch_dim=1, shape=(), **kwargs
    ):
        return (
            key,
            abstract_arrays.ShapedArray(
                shape=(batch_dim, *shape), dtype=jnp.float32
            ),
        )

    def sample(self, key, minval, maxval, **kwargs):
        key, sub_key = jax.random.split(key)
        v = jax.random.uniform(sub_key, minval=minval, maxval=maxval, **kwargs)
        return (key, v)

    def score(self, v, minval, maxval):
        return jnp.sum(jax.scipy.stats.uniform.logpdf(v, minval, maxval))

    # Pytree interfaces.
    def flatten(self):
        return (), ()

    def unflatten(self, data, xs):
        return Uniform()
