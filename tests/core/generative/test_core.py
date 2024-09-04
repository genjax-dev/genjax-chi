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

import genjax
from genjax import SelectionBuilder as S


class TestTupleAddr:
    def test_tupled_address(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ ("x", "x0")
            y = genjax.normal(x, 1.0) @ "y"
            return y

        tr = f.simulate(jax.random.PRNGKey(0), ())
        assert -2.7931314 == tr.project(jax.random.PRNGKey(1), S["x", "x0"])


class TestCombinators:
    """Tests for the generative function combinator methods."""

    def test_vmap(self):
        key = jax.random.PRNGKey(314159)

        @genjax.gen
        def model(x):
            v = genjax.normal(x, 1.0) @ "v"
            return genjax.normal(v, 0.01) @ "q"

        vmapped_model = model.vmap()

        jit_fn = jax.jit(vmapped_model.simulate)

        tr = jit_fn(key, (jnp.array([10.0, 20.0, 30.0]),))
        chm = tr.get_choices()

        varr = jnp.array([11.076874, 18.779837, 27.92488])
        qarr = jnp.array([11.096466, 18.75329, 27.932695])

        # The ellipsis syntax groups everything under a sub-key:
        assert jnp.array_equal(chm[..., "v"], varr)
        assert jnp.array_equal(chm[..., "q"], qarr)

        # check alternate access route:
        assert jnp.array_equal(chm(jnp.arange(3))["v"].unmask(), varr)

        assert jnp.array_equal(chm(jnp.arange(3))["q"].unmask(), qarr)

    def test_repeat(self):
        key = jax.random.PRNGKey(314159)

        @genjax.gen
        def model(x):
            return genjax.normal(x, 1.0) @ "x"

        vmap_model = model.vmap()
        repeat_model = model.repeat(n=3)

        vmap_tr = jax.jit(vmap_model.simulate)(key, (jnp.zeros(3),))
        repeat_tr = jax.jit(repeat_model.simulate)(key, (0.0,))

        repeatarr = jnp.array([1.0768734, -1.220163, -2.0751207])
        varr = jnp.array([1.0768734, -1.220163, -2.0751207])

        # Check that we get 3 repeated values:
        assert jnp.array_equal(repeat_tr.get_choices()[..., "x"], repeatarr)

        # check that the return value matches the traced values (in this case)
        assert jnp.array_equal(repeat_tr.get_retval(), repeatarr)

        # vmap does as well, but they are different due to internal seed splitting:
        assert jnp.array_equal(vmap_tr.get_choices()[..., "x"], varr)

    def test_or_else(self):
        key = jax.random.PRNGKey(314159)

        @genjax.gen
        def if_model(x):
            return genjax.normal(x, 1.0) @ "if_value"

        @genjax.gen
        def else_model(x):
            return genjax.normal(x, 5.0) @ "else_value"

        @genjax.gen
        def switch_model(toss: bool):
            return if_model.or_else(else_model)(toss, (1.0,), (10.0,)) @ "tossed"

        jit_fn = jax.jit(switch_model.simulate)
        if_tr = jit_fn(key, (True,))
        assert "if_value" in if_tr.get_choices()("tossed")

        else_tr = jit_fn(key, (False,))
        assert "else_value" in else_tr.get_choices()("tossed")
