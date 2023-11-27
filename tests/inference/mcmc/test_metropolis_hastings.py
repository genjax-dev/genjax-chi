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
from genjax import gen
from genjax import normal
from genjax import trace
from genjax import uniform
from genjax.inference.mcmc import MetropolisHastings


class TestMetropolisHastings:
    def test_simple_inf(self):
        @gen(genjax.Static)
        def normalModel(mu):
            x = trace("x", normal)(mu, 1.0)
            return x

        @gen(genjax.Static)
        def proposal(nowAt, d):
            current = nowAt["x"]
            x = trace("x", uniform)(current - d, current + d)
            return x

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(normalModel.simulate)(sub_key, (0.3,))
        mh = MetropolisHastings(proposal)
        for _ in range(0, 10):
            # Repeat the test for stochasticity.
            key, sub_key = jax.random.split(key)
            (new, check) = mh.apply(sub_key, tr, (0.25,))
            if check:
                assert tr.get_score() != new.get_score()
            else:
                assert tr.get_score() == new.get_score()

    def test_map_combinator(self):
        @genjax.gen(genjax.Static)
        def model():
            loc = genjax.normal(0.0, 1.0) @ "loc"
            xs = (
                genjax.Map(genjax.normal, in_axes=(None, 0))(loc, jnp.arange(10)) @ "xs"
            )
            return xs

        @genjax.gen(genjax.Static)
        def proposal(choices):
            loc = choices["loc"]
            xs = (
                genjax.Map(genjax.normal, in_axes=(None, 0))(loc, jnp.arange(10)) @ "xs"
            )
            return xs

        key = jax.random.PRNGKey(314159)
        trace = model.simulate(key, ())
        key, sub_key = jax.random.split(key)
        genjax.inference.mcmc.mh(proposal).apply(sub_key, trace, ())
        assert True
