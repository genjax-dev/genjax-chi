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
from genjax import Normal
from genjax._src.extras import blackjax
from genjax._src.language_decorator import gen


blackjax = blackjax()


@gen
def model():
    a = Normal(0.0, 1.0) @ "a"
    _ = Normal(0.0, 1.0) @ "b"
    _ = Normal(0.0, 1.0) @ "c"
    _ = Normal(0.0, 1.0) @ "d"
    _ = Normal(0.0, 1.0) @ "e"
    _ = Normal(0.0, 1.0) @ "f"
    _ = Normal(0.0, 1.0) @ "g"
    _ = Normal(0.0, 1.0) @ "h"
    _ = Normal(0.0, 1.0) @ "y"
    return a


def inf(key, init_trace):
    hmc = blackjax.nuts(genjax.select(["x"]), 0.01, jnp.ones(10), 500)
    key, _ = hmc.apply(init_trace)
    return key, None


class TestBlackJAXMicro:
    def test_benchmark(self, benchmark):
        key = jax.random.PRNGKey(314159)
        key, init_trace = jax.jit(model.importance)(key, ())
        jitted = jax.jit(inf)
        benchmark(jitted, key, init_trace)
