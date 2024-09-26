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
import pytest

import genjax
from genjax import ChoiceMapBuilder as C


class TestVmapCombinator:
    def test_vmap_combinator_simple_normal(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        map_score = tr.get_score()
        assert map_score == jnp.sum(tr.inner.get_score())

    def test_vmap_combinator_vector_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = jax.vmap(lambda idx, v: C[idx, "z"].set(v))(
            jnp.arange(3), jnp.array([3.0, 2.0, 3.0])
        )

        (_, w) = jax.jit(kernel.importance)(key, chm, (map_over,))
        assert (
            w
            == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]
            + genjax.normal.assess(C.v(2.0), (1.0, 1.0))[0]
            + genjax.normal.assess(C.v(3.0), (2.0, 1.0))[0]
        )

    def test_vmap_combinator_indexed_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = C[0, "z"].set(3.0)
        key, sub_key = jax.random.split(key)
        (_, w) = jax.jit(kernel.importance)(sub_key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]

        key, sub_key = jax.random.split(key)
        zv = jnp.array([3.0, -1.0, 2.0])
        chm = jax.vmap(lambda idx, v: C[idx, "z"].set(v))(jnp.arange(3), zv)
        (tr, _) = kernel.importance(sub_key, chm, (map_over,))
        for i in range(0, 3):
            v = tr.get_choices()[i, "z"]
            assert v.unmask() == zv[i]

    def test_vmap_combinator_nested_indexed_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def higher_model(x):
            return model(x) @ "outer"

        key = jax.random.PRNGKey(314159)
        map_over = jnp.ones((3, 3), dtype=float)
        chm = C[0, "outer", 1, "z"].set(1.0)
        (_, w) = jax.jit(higher_model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]

    def test_vmap_combinator_vmap_pytree(self):
        @genjax.vmap(in_axes=(None, (0, None)))
        @genjax.gen
        def foo(y, args):
            loc, (scale, _) = args
            x = genjax.normal(loc, scale) @ "x"
            return x + y

        key = jax.random.PRNGKey(314159)
        _ = jax.jit(foo.simulate)(key, (10.0, (jnp.arange(3.0), (1.0, jnp.arange(3)))))

    def test_vmap_combinator_assess(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        sample = tr.get_sample()
        map_score = tr.get_score()
        assert model.assess(sample, (map_over,))[0] == map_score

    def test_vmap_validation(self):
        @genjax.gen
        def foo(loc: float, scale: float):
            return genjax.normal(loc, scale) @ "x"

        key = jax.random.PRNGKey(314159)

        with pytest.raises(
            ValueError,
            match="vmap was requested to map its argument along axis 0, which implies that its rank should be at least 1, but is only 0",
        ):
            foo.vmap(in_axes=(0, None)).simulate(key, (10.0, jnp.arange(3.0)))

        # in_axes doesn't match args
        with pytest.raises(
            ValueError,
            match="vmap in_axes specification must be a tree prefix of the corresponding value",
        ):
            foo.vmap(in_axes=(0, (0, None))).simulate(key, (10.0, jnp.arange(3.0)))

        with pytest.raises(
            ValueError,
            match="vmap got inconsistent sizes for array axes to be mapped",
        ):
            foo.vmap(in_axes=0).simulate(key, (jnp.arange(2), jnp.arange(3)))

        # in_axes doesn't match args
        with pytest.raises(
            TypeError,
            match="Found incompatible dtypes, <class 'numpy.float32'> and <class 'numpy.int32'>",
        ):
            foo.vmap(in_axes=(None, 0)).simulate(key, (10.0, jnp.arange(3)))
