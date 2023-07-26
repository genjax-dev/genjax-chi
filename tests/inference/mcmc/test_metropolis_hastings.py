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
from genjax import tfp_uniform
from genjax import trace
from genjax.inference.mcmc import MetropolisHastings


class TestMetropolisHastings:
    def test_simple_inf(self):
        @gen
        def normalModel(mu):
            x = trace("x", normal)(mu, 1.0)
            return x

        @gen
        def proposal(nowAt, d):
            current = nowAt["x"]
            x = trace("x", tfp_uniform)(current - d, current + d)
            return x

        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(normalModel.simulate)(key, (0.3,))
        mh = MetropolisHastings(proposal)
        for _ in range(0, 10):
            # Repeat the test for stochasticity.
            key, (new, check) = mh.apply(key, tr, (0.25,))
            if check:
                assert tr.get_score() != new.get_score()
            else:
                assert tr.get_score() == new.get_score()

    def test_map_combinator(self):
        @genjax.gen
        def model():
            loc = genjax.normal(0.0, 1.0) @ "loc"
            xs = (
                genjax.Map(genjax.normal, in_axes=(None, 0))(loc, jnp.arange(10)) @ "xs"
            )
            return xs

        @genjax.gen
        def proposal(choices):
            loc = choices["loc"]
            xs = (
                genjax.Map(genjax.normal, in_axes=(None, 0))(loc, jnp.arange(10)) @ "xs"
            )
            return xs

        key = jax.random.PRNGKey(314159)
        key, trace = genjax.simulate(model)(key, ())
        genjax.inference.mcmc.mh(proposal).apply(key, trace, ())
        assert True
