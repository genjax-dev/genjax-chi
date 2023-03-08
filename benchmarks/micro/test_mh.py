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

import genjax
from genjax import MetropolisHastings
from genjax import normal
from genjax import tfp_uniform
from genjax import trace
from genjax._src.language_decorator import gen


@gen
def normalModel():
    x = trace("x", normal)(0.0, 1.0)
    return x


@gen
def proposal(nowAt, d):
    current = nowAt["x"]
    x = trace("x", tfp_uniform)(current - d, current + d)
    return x


def inf(key, init_trace):
    mh = MetropolisHastings(genjax.select(["x"]), proposal)

    def _inner(carry, xs):
        key, nowAt = carry
        key, (nowAt, _) = mh.apply(key, nowAt, (0.25,))
        return (key, nowAt), ()

    (key, nowAt), _ = jax.lax.scan(_inner, (key, init_trace), None, length=9)
    score = nowAt.get_score()
    return key, score


class TestMHMicro:
    def test_benchmark(self, benchmark):
        key = jax.random.PRNGKey(314159)
        key, init_trace = jax.jit(normalModel.simulate)(key, ())
        jitted = jax.jit(inf)
        jitted(key, init_trace)

        def _benchmark(key, iters=9):
            for _ in range(0, iters):
                key, score = jitted(key, init_trace)
                score.block_until_ready()  # force.

        benchmark(_benchmark, key)
