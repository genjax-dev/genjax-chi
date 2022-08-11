# Copyright 2022 MIT ProbComp
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


class Bernoulli:
    def abstract_eval(self, key, p, shape=()):
        return (key, abstract_arrays.ShapedArray(shape=shape, dtype=bool))

    def sample(self, key, p, **kwargs):
        key, sub_key = jax.random.split(key)
        v = jax.random.bernoulli(key, p, **kwargs)
        return (key, v)

    def score(self, v, p):
        return jnp.sum(jax.scipy.stats.bernoulli.logpmf(v, p))
