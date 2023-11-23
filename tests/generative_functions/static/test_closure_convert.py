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


class TestClosureConvert:
    def test_closure_convert(self):
        def emits_cc_gen_fn(v):
            @genjax.gen(genjax.Static)
            @genjax.dynamic_closure(v)
            def model(v):
                x = genjax.normal(jnp.sum(v), 1.0) @ "x"
                return x

            return model

        @genjax.gen(genjax.Static)
        def model():
            x = jnp.ones(5)
            gen_fn = emits_cc_gen_fn(x)
            v = gen_fn() @ "x"
            return (v, gen_fn)

        key = jax.random.PRNGKey(314159)
        _ = jax.jit(genjax.simulate(model))(key, ())
        assert True
