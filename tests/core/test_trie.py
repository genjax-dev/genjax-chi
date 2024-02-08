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

        assert list(t.address_sequence()) == [
            ("a", "a1"),
            ("a", "a2", "a21"),
            ("a", "a2", "a22"),
            ("b", "b1", "b11"),
            ("b", "b2"),
            ("c",),
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
        assert list(t.address_sequence()) == []

    def test_flat(self):
        assert list(
            trie_from_dict({"a": 1, "b": 2, "c": 3, "d": 4}).address_sequence()
        ) == [
            ("a",),
            ("b",),
            ("c",),
            ("d",),
        ]

    def test_nested(self):
        assert list(
            trie_from_dict({"a": {"b": {"c": {"d": 1}}}}).address_sequence()
        ) == [("a", "b", "c", "d")]
