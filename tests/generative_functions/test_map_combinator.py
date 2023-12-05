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


class TestMapCombinator:
    def test_map_simple_normal(self):
        @genjax.lang(genjax.Map, in_axes=(0,))
        @genjax.lang(genjax.Static)
        def model(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        map_score = tr.get_score()
        assert map_score == jnp.sum(tr.inner.get_score())

    def test_map_vector_choice_map_importance(self):
        @genjax.lang(genjax.Static)
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        model = genjax.Map(kernel, in_axes=(0,))
        map_over = jnp.arange(0, 3, dtype=float)
        chm = genjax.vector_choice_map(
            genjax.choice_map({"z": jnp.array([3.0, 2.0, 3.0])})
        )

        (tr, w) = jax.jit(model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.logpdf(3.0, 0.0, 1.0) + genjax.normal.logpdf(
            2.0, 1.0, 1.0
        ) + genjax.normal.logpdf(3.0, 2.0, 1.0)

    def test_map_indexed_choice_map_importance(self):
        @genjax.lang(genjax.Static)
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        model = genjax.Map(kernel, in_axes=(0,))
        map_over = jnp.arange(0, 3, dtype=float)
        chm = genjax.indexed_choice_map(
            [0],
            genjax.choice_map({"z": jnp.array([3.0])}),
        )
        key, sub_key = jax.random.split(key)
        (_, w) = jax.jit(model.importance)(sub_key, chm, (map_over,))
        assert w == genjax.normal.logpdf(3.0, 0.0, 1.0)

        key, sub_key = jax.random.split(key)
        zv = jnp.array([3.0, -1.0, 2.0])
        chm = genjax.indexed_choice_map([0, 1, 2], genjax.choice_map({"z": zv}))
        (tr, _) = model.importance(sub_key, chm, (map_over,))
        for i in range(0, 3):
            assert tr[i, "z"] == zv[i]

    def test_map_nested_indexed_choice_map_importance(self):
        @genjax.lang(genjax.Map, in_axes=(0,))
        @genjax.lang(genjax.Static)
        def model(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        @genjax.lang(genjax.Map, in_axes=(0,))
        @genjax.lang(genjax.Static)
        def higher_model(x):
            return model(x) @ "outer"

        key = jax.random.PRNGKey(314159)
        map_over = jnp.ones((3, 3), dtype=float)
        chm = genjax.indexed_choice_map(
            [0], {"outer": genjax.indexed_choice_map([1], {"z": jnp.array([[1.0]])})}
        )
        (_, w) = jax.jit(higher_model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.logpdf(1.0, 1.0, 1.0)

    def test_map_vmap_pytree(self):
        @genjax.lang(genjax.Static)
        def foo(y, args):
            loc, scale = args
            x = genjax.normal(loc, scale) @ "x"
            return x + y

        key = jax.random.PRNGKey(314159)
        _ = jax.jit(genjax.Map(foo, in_axes=(None, (0, None))).simulate)(
            key, (10.0, (jnp.arange(3.0), 1.0))
        )
