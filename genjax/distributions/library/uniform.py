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
from genjax.distributions.distribution_trace import DistributionTrace
from genjax.distributions.value_choice_map import ValueChoiceMap
from genjax.core.datatypes import GenerativeFunction


class _Uniform(GenerativeFunction):
    def abstract_eval(self, key, *params, shape=(), **kwargs):
        return (
            key,
            abstract_arrays.ShapedArray(shape=shape, dtype=jnp.float32),
        )

    def simulate(self, key, args, **kwargs):
        minval = args[0]
        maxval = args[1]
        key, sub_key = jax.random.split(key)
        v = jax.random.uniform(sub_key, minval=minval, maxval=maxval, **kwargs)
        score = jnp.sum(jax.scipy.stats.uniform.logpdf(v, minval, maxval))
        chm = ValueChoiceMap(v)
        tr = DistributionTrace(
            Uniform,
            args,
            chm,
            score,
        )
        return key, tr

    def importance(self, key, chm, args, **kwargs):
        minval = args[0]
        maxval = args[1]
        v = chm.get_value()
        weight = jnp.sum(jax.scipy.stats.uniform.logpdf(v, minval, maxval))
        tr = DistributionTrace(
            Uniform,
            args,
            chm,
            weight,
        )
        return key, (weight, tr)

    # Pytree interfaces.
    def flatten(self):
        return (), ()

    def unflatten(self, data, xs):
        return _Uniform()


Uniform = _Uniform()
