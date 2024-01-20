# Copyright 2023 MIT Probabilistic Computing Project
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
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.generative_functions.distributions.distribution import ExactDensity


class Dirac(JAXGenerativeFunction, ExactDensity):
    def sample(self, key, v):
        return v

    def logpdf(self, v1, v2):
        check = jnp.all(
            jnp.array(jtu.tree_leaves(jtu.tree_map(lambda v1, v2: v1 == v2)))
        )
        return jax.lax.cond(
            check,
            lambda _: 0.0,
            lambda _: -jnp.inf,
        )


dirac = Dirac()
