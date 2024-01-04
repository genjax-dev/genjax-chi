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
import jax.tree_util as jtu
import pytest


class TestVectorChoiceMap:
    def test_vector_choice_map_construction(self):
        chm = genjax.choice_map({"z": jnp.array([3.0])})
        v_chm = genjax.vector_choice_map(chm)
        assert v_chm.has_submap((0, "z"))


class TestIndexChoiceMap:
    def test_indexed_choice_map_construction(self):
        chm = genjax.indexed_choice_map([0], genjax.choice_map({"z": jnp.array([3.0])}))
        assert chm.has_submap((0, "z"))

        with pytest.raises(Exception):
            chm = genjax.indexed_choice_map(
                0, genjax.choice_map({"z": jnp.array([3.0])})
            )

        with pytest.raises(Exception):
            chm = genjax.indexed_choice_map(
                [0], genjax.choice_map({"z": jnp.array(3.0)})
            )

    def test_nested_indexed_choice_map_construction(self):
        inner = genjax.indexed_choice_map(
            jnp.arange(5),
            genjax.choice_map({"x": jnp.ones((5,))}),
        )
        outer = genjax.indexed_choice_map(
            [1], jtu.tree_map(lambda v: jnp.expand_dims(v, axis=0), inner)
        )
        assert outer[1].match(
            lambda: False, lambda v: jnp.all(v.inner["x"] == jnp.ones(5))
        )

    def test_indexed_choice_map_has_submap(self):
        chm = genjax.indexed_choice_map(
            [0, 3], genjax.choice_map({"z": jnp.array([3.0, 5.0])})
        )
        assert chm.has_submap((0, "z"))
        assert chm.has_submap((3, "z"))

    def test_indexed_choice_map_get_submap(self):
        chm = genjax.indexed_choice_map(
            [0, 3], genjax.choice_map({"z": jnp.array([3.0, 5.0])})
        )
        st = chm.get_submap((2, "x"))
        assert st == genjax.EmptyChoice()
        # When index is not available, always returns the first index slice inside of a Mask with a False flag.
        st = chm.get_submap((2, "z"))
        assert isinstance(st, genjax.Mask)
        assert st.mask is False


class TestVectorTrace:
    def test_vector_trace_static_selection(self):
        @genjax.Map(in_axes=(0,))
        @genjax.Static
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        key, sub_key = jax.random.split(key)
        vec_tr = jax.jit(kernel.simulate)(sub_key, (map_over,))
        sel = genjax.select("z")
        assert vec_tr.get_score() == vec_tr.project(sel)

    def test_vector_trace_index_selection(self):
        # Example generated using Map.
        @genjax.Map(in_axes=(0,))
        @genjax.Static
        def model(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        key, sub_key = jax.random.split(key)
        vec_tr = jax.jit(model.simulate)(sub_key, (map_over,))
        sel = genjax.indexed_select(jnp.array([1]), genjax.select("z"))
        score = genjax.normal.logpdf(vec_tr.get_choices()[1, "z"], map_over[1], 1.0)
        assert score == vec_tr.project(sel)

        # Example generated using Unfold.
        @genjax.Unfold(max_length=10)
        @genjax.Static
        def chain(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key, sub_key = jax.random.split(key)
        vec_tr = jax.jit(chain.simulate)(sub_key, (5, 0.0))
        sel = genjax.indexed_select([0], genjax.select("z"))
        proj_score = vec_tr.project(sel)
        latent_z_0 = vec_tr.filter(sel)[0, "z"].unsafe_unmask()
        z_score = genjax.normal.logpdf(latent_z_0, 0.0, 1.0)
        assert proj_score == z_score

        sel = genjax.indexed_select([1], genjax.select("z"))
        proj_score = vec_tr.project(sel)
        latent_z_1 = vec_tr.filter(sel)[1, "z"].unsafe_unmask()
        z_score = genjax.normal.logpdf(latent_z_1, latent_z_0, 1.0)
        assert proj_score == pytest.approx(z_score, 0.0001)

        sel = genjax.indexed_select([2], genjax.select("z"))
        proj_score = vec_tr.project(sel)
        latent_z_2 = vec_tr.filter(sel)[2, "z"].unsafe_unmask()
        z_score = genjax.normal.logpdf(latent_z_2, latent_z_1, 1.0)
        assert proj_score == z_score

        @genjax.Unfold(max_length=10)
        @genjax.Static
        def two_layer_chain(z):
            z1 = genjax.trace("z1", genjax.normal)(z, 1.0)
            _ = genjax.trace("z2", genjax.normal)(z1, 1.0)
            return z1

        key, sub_key = jax.random.split(key)
        vec_tr = jax.jit(two_layer_chain.simulate)(sub_key, (5, 0.0))
        sel = genjax.indexed_select([0], genjax.select("z1"))
        proj_score = vec_tr.project(sel)
        latent_z_1 = vec_tr.filter(sel)[0, "z1"].unsafe_unmask()
        z_score = genjax.normal.logpdf(latent_z_1, 0.0, 1.0)
        assert proj_score == z_score

        z1_sel = genjax.indexed_select([0], genjax.select("z1"))
        z2_sel = genjax.indexed_select([0], genjax.select("z2"))
        proj_score = vec_tr.project(z2_sel)
        latent_z_1 = vec_tr.filter(z1_sel)[0, "z1"].unsafe_unmask()
        latent_z_2 = vec_tr.filter(z2_sel)[0, "z2"].unsafe_unmask()
        z_score = genjax.normal.logpdf(latent_z_2, latent_z_1, 1.0)
        assert proj_score == z_score
