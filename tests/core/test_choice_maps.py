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


from genjax import ChoiceMap, Selection
from genjax import ChoiceMapBuilder as C
from genjax import SelectionBuilder as S


class TestSelections:
    def test_selection(self):
        new = S["x"] | S["z", "y"]
        assert new["x"]
        assert new["z", "y"]
        assert new["z", "y", "tail"]

        new = S["x"]
        assert new["x"]
        assert new["x", "y"]
        assert new["x", "y", "z"]
        new = S["x", "y", "z"]
        assert new["x", "y", "z"]
        assert not new["x"]
        assert not new["x", "y"]

    def test_selection_all(self):
        all_sel = Selection.all()

        assert all_sel == ~~all_sel
        assert all_sel["x"]
        assert all_sel["y", "z"]
        assert all_sel[()]

    def test_selection_none(self):
        none_sel = Selection.none()

        assert none_sel == ~~none_sel
        assert not none_sel["x"]
        assert not none_sel["y", "z"]
        assert not none_sel[()]

    def test_selection_complement(self):
        sel = S["x"] | S["y"]
        comp_sel = ~sel
        assert not comp_sel["x"]
        assert not comp_sel["y"]
        assert comp_sel["z"]

        # Test optimization: ~AllSel() = NoneSel()
        all_sel = Selection.all()
        assert ~all_sel == Selection.none()

        # Test optimization: ~NoneSel() = AllSel()
        none_sel = Selection.none()
        assert ~none_sel == Selection.all()

    def test_selection_and(self):
        sel1 = S["x"] | S["y"]
        sel2 = S["y"] | S["z"]
        and_sel = sel1 & sel2
        assert not and_sel["x"]
        assert and_sel["y"]
        assert not and_sel["z"]

        # Test optimization: AllSel() & other = other
        all_sel = Selection.all()
        assert (all_sel & sel1) == sel1
        assert (sel1 & all_sel) == sel1

        # Test optimization: NoneSel() & other = NoneSel()
        none_sel = Selection.none()
        assert (none_sel & sel1) == none_sel
        assert (sel1 and none_sel) == none_sel

    def test_selection_or(self):
        sel1 = S["x"]
        sel2 = S["y"]
        or_sel = sel1 | sel2
        assert or_sel["x"]
        assert or_sel["y"]
        assert not or_sel["z"]

        # Test optimization: AllSel() | other = AllSel()
        all_sel = Selection.all()
        assert (all_sel | sel1) == all_sel
        assert (sel1 | all_sel) == all_sel

        # Test optimization: NoneSel() | other = other
        none_sel = Selection.none()
        assert (none_sel | sel1) == sel1
        assert (sel1 | none_sel) == sel1

    def test_selection_mask(self):
        from genjax._src.core.interpreters.staging import Flag

        sel = S["x"] | S["y"]
        masked_sel = sel.mask(Flag(True))
        assert masked_sel["x"]
        assert masked_sel["y"]
        assert not masked_sel["z"]

        masked_sel = sel.mask(Flag(False))
        assert not masked_sel["x"]
        assert not masked_sel["y"]
        assert not masked_sel["z"]

    def test_selection_combination(self):
        sel1 = S["x"] | S["y"]
        sel2 = S["y"] | S["z"]
        combined_sel = (sel1 & sel2) | S["w"]
        assert not combined_sel["x"]
        assert combined_sel["y"]
        assert not combined_sel["z"]
        assert combined_sel["w"]

    def test_selection_ellipsis(self):
        # Create a selection with nested structure
        sel = S["a", "b", "c"] | S["x", "y", "z"]

        # Test that ... gives a free pass to one level of matching
        assert sel["a", ..., ...]
        assert sel["x", ..., ...]
        assert sel["a", ..., "c"]
        assert sel["x", ..., "z"]
        assert not sel["a", ..., "z"]

        assert not sel[...]
        assert not sel["a", "z", ...]

    def test_static_sel(self):
        xy_sel = Selection.at["x", "y"]
        assert not xy_sel[()]
        assert xy_sel["x", "y"]
        assert not xy_sel[0]
        assert not xy_sel["other_address"]

        # Test nested StaticSel
        nested_true_sel = Selection.at["x"].extend("y")
        assert nested_true_sel["y", "x"]
        assert not nested_true_sel["y"]

    def test_chm_sel(self):
        # Create a ChoiceMap
        chm = C["x", "y"].set(3.0) ^ C["z"].set(5.0)

        # Create a ChmSel from the ChoiceMap
        chm_sel = chm.get_selection()

        # Test selections
        assert chm_sel["x", "y"]
        assert chm_sel["z"]
        assert not chm_sel["w"]

        # Test nested selections
        assert chm_sel("x")["y"]

        # Test with empty ChoiceMap
        empty_chm = ChoiceMap.empty()
        empty_sel = empty_chm.get_selection()
        assert empty_sel == Selection.none()


class TestChoiceMapBuilder:
    def test_set(self):
        chm = C["a", "b"].set(1)
        assert chm["a", "b"] == 1

        # membership only returns True for the actual path
        assert ("a", "b") in chm
        assert "a" not in chm
        assert "b" in chm("a")

    def test_nested_set(self):
        chm = C["x"].set(C["y"].set(2))
        assert chm["x", "y"] == 2
        assert ("x", "y") in chm
        assert "y" not in chm

    def test_empty(self):
        assert C.n() == ChoiceMap.empty()

        # n() at any level returns the empty choice map
        assert C["x", "y"].n() == ChoiceMap.empty()

    def test_v_matches_set(self):
        assert C["a", "b"].set(1) == C["a", "b"].v(1)

        inner = C["y"].v(2)

        # .v on a ChoiceMap wraps the choicemap in ValueChm (not advisable!)
        assert C["x"].v(inner)("x").get_value() == inner

    def test_from_mapping(self):
        mapping = [("a", 1.0), (("b", "c"), 2.0), (("b", "d", "e"), {"f": 3.0})]
        chm = C["base"].from_mapping(mapping)
        assert chm["base", "a"] == 1
        assert chm["base", "b", "c"] == 2

        # dict entries work in from_mapping values
        assert chm["base", "b", "d", "e", "f"] == 3

        assert ("base", "a") in chm
        assert ("base", "b", "c") in chm
        assert ("b", "c") in chm("base")

    def test_d(self):
        chm = C["top"].d({
            "x": 3,
            "y": {"z": 4, "w": C["bottom"].d({"v": 5})},
        })
        assert chm["top", "x"] == 3

        # notice that dict values are converted into ChoiceMap.d calls
        assert chm["top", "y", "z"] == 4
        assert chm["top", "y", "w", "bottom", "v"] == 5

    def test_kw(self):
        chm = C["root"].kw(a=1, b=C["nested"].kw(c=2, d={"deep": 3}))
        assert chm["root", "a"] == 1
        assert chm["root", "b", "nested", "c"] == 2

        # notice that dict values are converted into chms
        assert chm["root", "b", "nested", "d", "deep"] == 3


class TestChoiceMap:
    # TODO add tests for the instance methods, so close here...
    # TODO go and make sure docs are correctly generated for everything

    def test_empty(self):
        empty_chm = ChoiceMap.empty()
        assert empty_chm.static_is_empty()

    def test_value(self):
        value_chm = ChoiceMap.value(42.0)
        assert value_chm.get_value() == 42.0
        assert value_chm.has_value()

        # NO sub-paths are inside a ValueChm.
        assert () in value_chm

    def test_kv(self):
        chm = ChoiceMap.kw(x=1, y=2)
        assert chm["x"] == 1
        assert chm["y"] == 2

        assert "x" in chm
        assert "y" in chm
        assert "other_value" not in chm

    def test_d(self):
        chm = ChoiceMap.d({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
        assert chm["a"] == 1
        assert chm["b", "c"] == 2
        assert chm["b", "d", "e"] == 3
        assert "a" in chm
        assert ("b", "c") in chm
        assert ("b", "d", "e") in chm

    def test_from_mapping(self):
        mapping = [("x", 1), (("y", "z"), 2), (("w", "v", "u"), 3)]
        chm = ChoiceMap.from_mapping(mapping)
        assert chm["x"] == 1
        assert chm["y", "z"] == 2
        assert chm["w", "v", "u"] == 3
        assert "x" in chm
        assert ("y", "z") in chm
        assert ("w", "v", "u") in chm
