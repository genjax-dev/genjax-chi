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
import jax.tree_util as jtu

import genjax
from genjax import normal
from genjax import tfp_uniform
from genjax import trace
from genjax import gen
from genjax.inference import smc
from genjax import index_choice_map, choice_map
from genjax import diff, NoChange, UnknownChange
from genjax import index_select


class TestSimpleSMC:
    def test_initialize_and_update(self):
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
        obs = index_choice_map(
            [1],
            choice_map({"x": jnp.array([1.0])}),
        )
        key, smc_state = smc.smc_update(
            key,
            smc_state,
            (diff(1, UnknownChange), diff(init_state, NoChange)),
            obs,
        )
        assert True

    def test_smc_with_scan(self):
        @gen(genjax.Unfold, max_length=10)
        def chain(z_prev):
            z = normal(z_prev, 1.0) @ "z"
            x = normal(z, 1.0) @ "x"
            return z

        obs = index_choice_map(
            [0, 1, 2, 3],
            choice_map({"x": jnp.array([1.0, 2.0, 3.0, 4.0])}),
        )

        def simple_smc(key, obs, init_state):
            index_sel = index_select(0)
            key, smc_state = smc.smc_initialize(key, chain, (0, init_state), obs, 100)
            obs = jtu.tree_map(lambda v: v[1:], obs)

            def _inner(carry, xs):
                key, smc_state, t = carry
                obs_slice = xs
                t = t + 1
                key, smc_state = smc.smc_update(
                    key,
                    smc_state,
                    (diff(t, UnknownChange), diff(init_state, NoChange)),
                    obs_slice,
                )
                return (key, smc_state, t), (smc_state, )

            (key, smc_state, _), (stacked, ) = jax.lax.scan(
                _inner,
                (key, smc_state, 0),
                obs,
            )
            return key, stacked

        key = jax.random.PRNGKey(314159)
        key, smc_state = simple_smc(key, obs, 0.0)

        assert True
