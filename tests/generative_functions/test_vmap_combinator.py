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

import genjax
import jax
import jax.numpy as jnp
from genjax import ChoiceMap as C


class TestVmapCombinator:
    def test_vmap_combinator_simple_normal(self):
        @genjax.vmap_combinator(in_axes=(0,))
        @genjax.static_gen_fn
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        map_score = tr.get_score()
        assert map_score == jnp.sum(tr.inner.get_score())

    def test_vmap_combinator_vector_choice_map_importance(self):
        @genjax.vmap_combinator(in_axes=(0,))
        @genjax.static_gen_fn
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = jax.vmap(lambda idx, v: C.n.at[idx, "z"].set(v))(
            jnp.arange(3), jnp.array([3.0, 2.0, 3.0])
        )

        (_, w, _) = jax.jit(kernel.importance)(key, chm, (map_over,))
        assert (
            w
            == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]
            + genjax.normal.assess(C.v(2.0), (1.0, 1.0))[0]
            + genjax.normal.assess(C.v(3.0), (2.0, 1.0))[0]
        )

    def test_vmap_combinator_indexed_choice_map_importance(self):
        @genjax.vmap_combinator(in_axes=(0,))
        @genjax.static_gen_fn
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = C.n.at[0, "z"].set(3.0)
        key, sub_key = jax.random.split(key)
        (_, w, _) = jax.jit(kernel.importance)(sub_key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]

        key, sub_key = jax.random.split(key)
        zv = jnp.array([3.0, -1.0, 2.0])
        chm = jax.vmap(lambda idx, v: C.n.at[idx, "z"].set(v))(jnp.arange(3), zv)
        (tr, _, _) = kernel.importance(sub_key, chm, (map_over,))
        for i in range(0, 3):
            v = tr.get_sample()[i, "z"]
            assert v == zv[i]

    def test_vmap_combinator_nested_indexed_choice_map_importance(self):
        @genjax.vmap_combinator(in_axes=(0,))
        @genjax.static_gen_fn
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        @genjax.vmap_combinator(in_axes=(0,))
        @genjax.static_gen_fn
        def higher_model(x):
            return model(x) @ "outer"

        key = jax.random.PRNGKey(314159)
        map_over = jnp.ones((3, 3), dtype=float)
        chm = C.n.at[0, "outer", 1, "z"].set(1.0)
        (_, w, _) = jax.jit(higher_model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]

    def test_vmap_combinator_vmap_pytree(self):
        @genjax.vmap_combinator(in_axes=(None, (0, None)))
        @genjax.static_gen_fn
        def foo(y, args):
            loc, scale = args
            x = genjax.normal(loc, scale) @ "x"
            return x + y

        key = jax.random.PRNGKey(314159)
        _ = jax.jit(foo.simulate)(key, (10.0, (jnp.arange(3.0), 1.0)))
