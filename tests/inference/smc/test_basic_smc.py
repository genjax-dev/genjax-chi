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

import genjax
from genjax import normal
from genjax import tfp_uniform
from genjax import trace
from genjax import gen
from genjax.inference import smc
from genjax import index_choice_map, choice_map


class TestSimpleSMC:
    def test_initialize(self):
        @gen(genjax.Unfold, max_length=10)
        def chain(z_prev):
            z = normal(z_prev, 1.0) @ "z"
            x = normal(z, 1.0) @ "x"
            return z

        key = jax.random.PRNGKey(314159)
        init_len = 0
        init_state = 0.0
        obs = index_choice_map(
            [0],
            choice_map({"x": jnp.array([1.0])}),
        )
        key, smc_state = smc.smc_initialize(
            key, chain, (init_len, init_state), obs, 100
        )
