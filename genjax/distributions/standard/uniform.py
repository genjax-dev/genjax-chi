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
import jax.core as jc
from dataclasses import dataclass
from genjax.core.tracetypes import RealInterval, Reals
from genjax.distributions.distribution import Distribution


@dataclass
class _Uniform(Distribution):
    def sample(self, key, minval, maxval, **kwargs):
        return jax.random.uniform(key, minval=minval, maxval=maxval, **kwargs)

    def logpdf(self, key, v, minval, maxval):
        return jnp.sum(jax.scipy.stats.uniform.logpdf(v, minval, maxval))

    def __trace_type__(self, key, minval, maxval, **kwargs):
        shape = kwargs.get("shape", ())
        if isinstance(minval, jc.ShapedArray) or isinstance(
            maxval, jc.ShapedArray
        ):
            return Reals(shape)
        else:
            return RealInterval(shape, minval, maxval)


Uniform = _Uniform()
