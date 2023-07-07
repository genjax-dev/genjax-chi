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
import pytest

import genjax

class TestVectorChoiceMap:
    def test_vector_choice_map_construction(self):
        chm = genjax.choice_map({"z": jnp.array([3.0])})
        v_chm = genjax.vector_choice_map(chm)
        assert v_chm.has_subtree("z")

class TestIndexChoiceMap:
    def test_index_choice_map_construction(self):
        chm = genjax.index_choice_map(genjax.choice_map({"z": jnp.array([3.0])}), [0])
        assert chm.has_subtree((0, "z"))

        with pytest.raises(Exception):
            chm = genjax.index_choice_map(genjax.choice_map({"z": jnp.array([3.0])}), 0)
            
        with pytest.raises(Exception):
            chm = genjax.index_choice_map(genjax.choice_map({"z": jnp.array(3.0)}), [0])

    def test_index_choice_map_has_subtree(self):
            chm = genjax.index_choice_map(genjax.choice_map({"z": jnp.array([3.0, 5.0])}), [0, 3])
            assert chm.has_subtree((0, "z"))
            assert chm.has_subtree((3, "z"))
    
    def test_index_choice_map_get_subtree(self):
            chm = genjax.index_choice_map(genjax.choice_map({"z": jnp.array([3.0, 5.0])}), [0, 3])
            st = chm.get_subtree((2, "x"))
            assert st == genjax.EmptyChoiceMap()
            # When index is not available, always returns the first index slice inside of a BooleanMask with a False flag.
            st = chm.get_subtree((2, "z"))
            assert isinstance(st, genjax.ValueChoiceMap)
            assert isinstance(st.get_leaf_value(), genjax.BooleanMask)
            assert st.get_leaf_value().mask == False
            assert st.get_leaf_value().value == 3.0


class TestVectorTrace:
    def test_vector_trace_builtin_selection(self):
        key = jax.random.PRNGKey(314159)
        
        @genjax.gen
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z


        model = genjax.Map(kernel, in_axes=(0,))

        map_over = jnp.arange(0, 50)
        key, vec_tr = jax.jit(genjax.simulate(model))(key, (map_over,))
        sel = genjax.select("z")
        assert vec_tr.get_score() == vec_tr.project(sel)
    
    def test_vector_trace_index_selection(self):
        key = jax.random.PRNGKey(314159)
        
        @genjax.gen
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z


        model = genjax.Map(kernel, in_axes=(0,))

        map_over = jnp.arange(0, 50)
        key, vec_tr = jax.jit(genjax.simulate(model))(key, (map_over,))
        sel = genjax.index_select(jnp.array([1]), genjax.select("z"))
        score = genjax.normal.logpdf(vec_tr.get_choices()["z"][1], 1.0, 1.0)
        assert score == vec_tr.project(sel)
