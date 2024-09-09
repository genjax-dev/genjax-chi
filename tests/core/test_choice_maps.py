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

    # def test_nested_set(self):
    #     chm = C["x"].set(C["y"].set(2))
    #     assert chm["x", "y"] == 2
    #     assert ("x", "y") in chm
    #     assert "y" not in chm

    # def test_call(self):
    #     chm = C["f"](1, 2, x=3)
    #     assert chm["f", "args", 0] == 1
    #     assert chm["f", "args", 1] == 2
    #     assert chm["f", "kwargs", "x"] == 3

    # def test_getitem(self):
    #     chm = C["list"][0].set(5)
    #     assert chm["list", 0] == 5
    #     assert "list" in chm
    #     assert 0 in chm["list"]

    # def test_combine_methods(self):
    #     chm = C["a"].set(1) ^ C["b"](2) ^ C["c"][0].set(3)
    #     assert chm["a"] == 1
    #     assert chm["b", "args", 0] == 2
    #     assert chm["c", 0] == 3

    # def test_extend(self):
    #     chm = C["x"].extend("y").set(4)
    #     assert chm["x", "y"] == 4
    #     assert "x" in chm
    #     assert "y" in chm["x"]

    # def test_complex_nesting(self):
    #     chm = C["a"].set(C["b"](C["c"][0].set(5)))
    #     assert chm["a", "b", "args", 0, "c", 0] == 5
    #     assert "a" in chm
    #     assert "b" in chm["a"]
    #     assert "args" in chm["a", "b"]
    #     assert 0 in chm["a", "b", "args"]
    #     assert "c" in chm["a", "b", "args", 0]
    #     assert 0 in chm["a", "b", "args", 0, "c"]


class TestChoiceMap:
    def test_value_map(self):
        value_chm = ChoiceMap.value(3.0)
        assert 3.0 == value_chm.get_value()
        assert () in value_chm

    def test_address_map(self):
        chm = C["x"].set(3.0)
        assert chm["x"] == 3.0
