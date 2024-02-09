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
import jax.random
from genjax._src.core.datatypes.generative import ChoiceValue
from genjax._src.core.datatypes.trie import Trie
from genjax._src.shortcuts import trie_from_dict


class TestTrie:
    def test_address_generation(self):
        d = {
            "a": {
                "a1": 1,
                "a2": {"a21": 21, "a22": 22},
            },
            "b": {
                "b1": {
                    "b11": 11,
                },
                "b2": 2,
            },
            "c": 3,
        }
        t = trie_from_dict(d)

        assert list(t.to_sequence()) == [
            (("a", "a1"), 1),
            (("a", "a2", "a21"), 21),
            (("a", "a2", "a22"), 22),
            (("b", "b1", "b11"), 11),
            (("b", "b2"), 2),
            (("c",), 3),
        ]

        assert t["a", "a1"] == ChoiceValue(1)
        assert t["a", "a2", "a21"] == ChoiceValue(21)
        assert t["a"]["a2"]["a21"] == ChoiceValue(21)
        assert t["a", "a2"]["a21"] == ChoiceValue(21)

        assert t["a"] == trie_from_dict(d["a"])
        assert t["b"] == trie_from_dict(d["b"])
        assert t["c"] == ChoiceValue(3)

    def test_empty(self):
        t = Trie()
        assert list(t.to_sequence()) == []

    def test_flat(self):
        assert list(trie_from_dict({"a": 1, "b": 2, "c": 3, "d": 4}).to_sequence()) == [
            (("a",), 1),
            (("b",), 2),
            (("c",), 3),
            (("d",), 4),
        ]

    def test_nested(self):
        assert list(trie_from_dict({"a": {"b": {"c": {"d": 1}}}}).to_sequence()) == [
            (("a", "b", "c", "d"), 1)
        ]

    def test_unfold(self):
        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def x(prev):
            return genjax.normal(prev, 0.1) @ "x"

        key = jax.random.PRNGKey(0)
        tr = x.simulate(key, (3, 5.0))

        assert list(tr.to_sequence()) == [
            ((0, "x"), 5.113846),
            ((1, "x"), 5.2523155),
            ((2, "x"), 5.3999386),
        ]

    def test_unfold_structure_below(self):
        @genjax.static_gen_fn
        def step():
            return 0.1 + 10.0 * (genjax.flip(0.5) @ "flip")

        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def x(prev):
            s = step() @ "step"
            return genjax.normal(prev, s) @ "x"

        key = jax.random.PRNGKey(0)
        tr = x.simulate(key, (3, 5.0))

        assert list(tr.to_sequence()) == [
            ((0, "step", "flip"), 0),
            ((1, "step", "flip"), 1),
            ((2, "step", "flip"), 0),
            ((0, "x"), 4.9019623),
            ((1, "x"), 1.8536623),
            ((2, "x"), 1.9565629),
        ]
