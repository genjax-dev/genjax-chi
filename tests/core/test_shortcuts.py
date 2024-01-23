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
import jax.numpy as jnp


class TestShortcuts:
    def test_no_args_gives_empty_map(self):
        assert genjax.choice_map() == genjax.HierarchicalChoiceMap()

    def test_bare_value_gives_choice_value(self):
        assert genjax.choice_map(3) == genjax.ChoiceValue(3)
        v1 = genjax.choice_map(jnp.array([1.0, 2.0]))
        v2 = genjax.ChoiceValue(jnp.array([1.0, 2.0]))
        assert isinstance(v1, genjax.ChoiceValue) and (v1.value == v2.value).all()

    def test_choice_map_from_dict(self):
        hcm = genjax.HierarchicalChoiceMap()
        hcm = hcm.insert("x", 3)
        hcm = hcm.insert("y", 4)
        assert genjax.choice_map({"x": 3, "y": 4}) == hcm

    def test_hierarchical_choice_map_from_dict(self):
        hcm = genjax.HierarchicalChoiceMap()
        hcm = hcm.insert(("x", "xa"), 3)
        hcm = hcm.insert(("x", "xb"), 4)
        hcm = hcm.insert(("y", "ya"), 5)
        hcm = hcm.insert(("y", "yb"), 6)
        assert (
            genjax.choice_map(
                {
                    "x": {"xa": 3, "xb": 4},
                    "y": {"ya": 5, "yb": 6},
                }
            )
            == hcm
        )

    def test_indexed_choice_map(self):
        icm1 = genjax.indexed_choice_map(
            jnp.array([1, 2, 3]), genjax.choice_map({"x": jnp.array([10, 20, 30])})
        )
        icm2 = genjax.choice_map({1: 10, 2: 20, 3: 30})
        for j in range(1, 4):
            for m in [icm1, icm2]:
                assert m.has_submap((j, "x"))
                assert m[(j, "x")].unmask() == 10 * j
