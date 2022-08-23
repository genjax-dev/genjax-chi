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


class _Normal(GenerativeFunction):
    def abstract_eval(self, key, shape=()):
        return (
            key,
            abstract_arrays.ShapedArray(shape=shape, dtype=jnp.float32),
        )

    def simulate(self, key, args, **kwargs):
        key, sub_key = jax.random.split(key)
        v = jax.random.normal(sub_key, **kwargs)
        chm = ValueChoiceMap(v)
        score = jax.scipy.stats.norm.logpdf(v)
        tr = DistributionTrace(
            Normal,
            args,
            chm,
            score,
        )
        return key, tr

    def importance(self, key, chm, args, **kwargs):
        v = chm.get_value()
        weight = jnp.sum(jax.scipy.stats.norm.logpdf(v))
        tr = DistributionTrace(
            Normal,
            args,
            chm,
            weight,
        )
        return key, (weight, tr)

    def diff(self, key, prev, new, args, **kwargs):
        bwd = prev.get_score()
        v = new.get_value()
        fwd = jnp.sum(jax.scipy.stats.norm.logpdf(v))
        return key, (fwd - bwd, (v,))

    def update(self, key, prev, new, args, **kwargs):
        bwd = prev.get_score()
        v = new.get_value()
        chm = ValueChoiceMap(v)
        fwd = jnp.sum(jax.scipy.stats.norm.logpdf(v))
        tr = DistributionTrace(
            Normal,
            args,
            chm,
            fwd,
        )
        return key, (fwd - bwd, tr, prev.get_choices())

    def flatten(self):
        return (), ()

    def unflatten(self, values, slices):
        return _Normal()


Normal = _Normal()
