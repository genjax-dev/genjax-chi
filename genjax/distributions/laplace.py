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


class Laplace:
    def abstract_eval(self, key, shape=()):
        return (key, abstract_arrays.ShapedArray(shape=shape, dtype=float))

    def sample(self, key, **kwargs):
        key, sub_key = jax.random.split(key)
        v = jax.random.laplace(key, **kwargs)
        return (key, v)

    def score(self, v):
        return jnp.sum(jax.scipy.stats.laplace.logpdf(v))
