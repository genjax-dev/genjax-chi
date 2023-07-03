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


@genjax.gen
def kernel(x):
    z = genjax.trace("z", genjax.normal)(x, 1.0)
    return z


model = genjax.Map(kernel, in_axes=(0,))


@genjax.gen
def proposal(x):
    z = genjax.normal(x, 1.0) @ (0, "z")
    return z


@genjax.gen
def foo(y, args):
    loc, scale = args
    x = genjax.normal(loc, scale) @ "x"
    return x + y


class TestMapCombinator:
    def test_map_simple_normal(self):
        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 50)
        key, tr = jax.jit(genjax.simulate(model))(key, (map_over,))
        map_score = tr.get_score()
        assert map_score == jnp.sum(tr.inner.get_score())

    def test_map_vector_choice_map_importance(self):
        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 3)
        chm = genjax.vector_choice_map(
            genjax.choice_map({"z": jnp.array([3.0, 2.0, 3.0])})
        )
        key, (w, tr) = jax.jit(model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.logpdf(3.0, 0.0, 1.0) + genjax.normal.logpdf(
            2.0, 1.0, 1.0
        ) + genjax.normal.logpdf(3.0, 2.0, 1.0)

    def test_map_index_choice_map_importance(self):
        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 3)
        chm = genjax.index_choice_map(genjax.choice_map({"z": jnp.array([3.0])}), [0])
        key, (w, _) = jax.jit(model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.logpdf(3.0, 0.0, 1.0)

    def test_map_vmap_pytree(self):
        key = jax.random.PRNGKey(314159)
        _, trace = genjax.simulate(genjax.Map(foo, in_axes=(None, (0, None))))(
            key, (10.0, (jnp.arange(3.0), 1.0))
        )
