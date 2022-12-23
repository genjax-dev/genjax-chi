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
import numpy as np

from genjax._src.core.tracetypes import Finite
from genjax._src.generative_functions.distributions.distribution import (
    ExactDensity,
)


@dataclass
class _Categorical(ExactDensity):
    def sample(self, key, logits, **kwargs):
        return jax.random.categorical(key, logits, **kwargs)

    def logpdf(self, v, logits, **kwargs):
        axis = kwargs.get("axis", -1)
        logpdf = jnp.log(jax.nn.softmax(logits, axis=axis))
        w = jnp.sum(logpdf[v])
        return w

    def get_trace_type(self, key, logits, **kwargs):
        shape = kwargs.get("shape", ())
        return Finite(np.prod(logits.shape), shape)


Categorical = _Categorical()
