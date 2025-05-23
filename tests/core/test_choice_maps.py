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


import re

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import genjax
from genjax import ChoiceMap, Selection
from genjax import ChoiceMapBuilder as C
from genjax import SelectionBuilder as S
from genjax._src.core.generative.choice_map import (
    ChoiceMapNoValueAtAddress,
    Static,
    StaticAddress,
    Switch,
)
from genjax._src.core.generative.functional_types import Mask
from genjax._src.core.typing import Any


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

    def test_wildcard_selection(self):
        sel = S["x"] | S[..., "y"]

        assert sel["x"]
        assert sel["any_address", "y"]
        assert sel["rando", "y", "tail"]

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

        # none can't be extended
        assert Selection.none().extend("a", "b") == Selection.none()

    def test_selection_builder_properties(self):
        # Test S.all
        assert S.all == Selection.all()
        assert S.all["x"]
        assert S.all["y", "z"]
        assert S.all[()]

        # Test S.none
        assert S.none == Selection.none()
        assert not S.none["x"]
        assert not S.none["y", "z"]
        assert not S.none[()]

        # Test S.leaf
        leaf_sel = S.leaf
        assert leaf_sel == Selection.leaf()
        leaf_sel = leaf_sel.extend("a", "b")
        assert leaf_sel["a", "b"]
        assert not leaf_sel["a"]
        assert not leaf_sel["a", "b", "c"]

        # Test empty tuple selection
        assert S[()] == Selection.leaf()
        assert () in S[()]

    def test_selection_leaf(self):
        leaf_sel = Selection.leaf().extend("x", "y")
        assert not leaf_sel["x"]
        assert leaf_sel["x", "y"]

        # only exact matches are allowed
        assert not leaf_sel["x", "y", "z"]

        # wildcards are not allowed
        with pytest.raises(TypeError):
            leaf_sel[..., "y"]  # pyright: ignore

    def test_selection_complement(self):
        sel = S["x"] | S["y"]
        comp_sel = ~sel
        assert not comp_sel["x"]
        assert not comp_sel["y"]
        assert comp_sel["z"]

        # Complement of a complement
        assert ~~sel == sel

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
        assert not and_sel.check()
        assert and_sel.get_subselection("y").check()
        assert not and_sel["z"]

        # Test optimization: AllSel() & other = other
        all_sel = Selection.all()
        assert (all_sel & sel1) == sel1
        assert (sel1 & all_sel) == sel1

        # Test optimization: NoneSel() & other = NoneSel()
        none_sel = Selection.none()
        assert (none_sel & sel1) == none_sel
        assert (sel1 & none_sel) == none_sel

        # idempotence
        assert sel1 & sel1 == sel1
        assert sel2 & sel2 == sel2

    def test_selection_or(self):
        sel1 = S["x"]
        sel2 = S["y"]
        or_sel = sel1 | sel2
        assert or_sel["x"]
        assert or_sel["y"]
        assert or_sel.get_subselection("y").check()
        assert not or_sel["z"]

        # Test optimization: AllSel() | other = AllSel()
        all_sel = Selection.all()
        assert (all_sel | sel1) == all_sel
        assert (sel1 | all_sel) == all_sel

        # Test optimization: NoneSel() | other = other
        none_sel = Selection.none()
        assert (none_sel | sel1) == sel1
        assert (sel1 | none_sel) == sel1

        # idempotence
        assert sel1 | sel1 == sel1
        assert sel2 | sel2 == sel2

    def test_selection_filter(self):
        # Create a ChoiceMap
        chm = ChoiceMap.kw(x=1, y=2, z=3)

        # Create a Selection
        sel = S["x"] | S["y"]

        # Filter the ChoiceMap using the Selection
        filtered_chm = sel.filter(chm)

        # Test that the filtered ChoiceMap contains only selected addresses
        assert "x" in filtered_chm
        assert "y" in filtered_chm
        assert "z" not in filtered_chm

        # Test values are preserved
        assert filtered_chm["x"] == 1
        assert filtered_chm["y"] == 2

        # Test with an empty Selection
        empty_sel = Selection.none()
        assert empty_sel.filter(chm).static_is_empty()

        # Test with an all-inclusive Selection
        all_sel = Selection.all()
        all_filtered_chm = all_sel.filter(chm)
        assert all_filtered_chm == chm

        # Test with a nested ChoiceMap
        nested_chm = ChoiceMap.kw(a={"b": 1, "c": 2}, d=3)
        nested_sel = S["a", "b"] | S["d"]
        nested_filtered_chm = nested_sel.filter(nested_chm)
        assert "d" in nested_filtered_chm
        assert "b" in nested_filtered_chm("a")
        assert "c" not in nested_filtered_chm("a")

    def test_selection_combination(self):
        sel1 = S["x"] | S["y"]
        sel2 = S["y"] | S["z"]
        combined_sel = (sel1 & sel2) | S["w"]
        assert not combined_sel["x"]
        assert combined_sel["y"]
        assert not combined_sel["z"]
        assert combined_sel["w"]

    def test_selection_contains(self):
        # Create a selection
        sel = S["x"] | S["y", "z"]

        # Test that __contains__ works like __getitem__
        assert "x" in sel
        assert sel["x"]
        assert ("y", "z") in sel
        assert sel["y", "z"]
        assert "y" not in sel
        assert not sel["y"]
        assert "w" not in sel
        assert not sel["w"]

        # Test with nested selections
        nested_sel = S["c"].extend("a", "b")

        assert ("a", "b", "c") in nested_sel
        assert nested_sel["a", "b", "c"]

        assert ("a", "b") not in nested_sel
        assert not nested_sel["a", "b"]

        # check works like __contains__
        assert not nested_sel("a")("b").check()
        assert nested_sel("a")("b")("c").check()

    def test_ellipsis_not_allowed(self):
        # Create a selection with nested structure
        sel = S["a", "b", "c"] | S["x", "y", "z"]

        with pytest.raises(TypeError):
            sel["a", ..., ...]  # pyright: ignore

    def test_static_sel(self):
        xy_sel = Selection.at["x", "y"]
        assert not xy_sel[()]
        assert xy_sel["x", "y"]
        assert not xy_sel["other_address"]

        # Test nested StaticSel
        nested_true_sel = Selection.at["x"].extend("y")
        assert nested_true_sel["y", "x"]
        assert not nested_true_sel["y"]

    def test_chm_sel(self):
        # Create a ChoiceMap
        chm = C["x", "y"].set(3.0) | C["z"].set(5.0)

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
        assert ChoiceMap.builder.set(1.0) == C[()].set(1.0)

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

    def test_update(self):
        chm = C["x", "y"].set(2)

        # update at a spot populated by a choicemap
        updated = chm.at["x"].update(lambda m: C["z"].set(m))
        assert updated["x", "z", "y"] == 2

        # update that hits a spot
        updated_choice = chm.at["x", "y"].update(jnp.square)
        assert updated_choice["x", "y"] == 4

        # update that hits an empty spot:
        updated_empty = chm.at["q"].update(lambda m: C["z"].set(m))
        assert updated_empty(("q", "z")).static_is_empty()

        # filling the spot is fine:
        updated_empty_2 = chm.at["q"].update(lambda m: C["z"].set(2))
        assert updated_empty_2["q", "z"] == 2

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

    def test_switch(self):
        chm1 = C["x"].set(1)
        chm2 = C["y"].set(2)
        chm3 = C["z"].set(3)

        # Test with integer index
        switched = C["root"].switch(1, [chm1, chm2, chm3])
        assert switched("root") == chm2

        # Test with array index
        idx = jnp.array(2)
        switched_array = C["root"].switch(idx, [chm1, chm2, chm3])

        # Can get values from any component, masked to the correct idx
        assert switched_array["root", "x"] == Mask(1, jnp.array(False))
        assert switched_array["root", "y"] == Mask(2, jnp.array(False))
        assert switched_array["root", "z"] == Mask(3, jnp.array(True))

        # Test nested switch
        nested = C["outer"].switch(
            0,
            [
                C["inner"].switch(1, [chm1, chm2, chm3]),
                C["inner"].switch(2, [chm1, chm2, chm3]),
            ],
        )

        assert nested("outer")("inner") == chm2

        # Test with empty choice maps
        empty_switch = C["root"].switch(0, [C.n(), C.n()])
        assert empty_switch.static_is_empty()


class TestChoiceMap:
    def test_empty(self):
        empty_chm = ChoiceMap.empty()
        assert empty_chm.static_is_empty()

    def test_choice(self):
        choice = ChoiceMap.choice(42.0)
        assert choice.get_value() == 42.0
        assert choice.has_value()

        # NO sub-paths are inside a ValueChm.
        assert () in choice

        # A choice with a mask that is concrete False is empty.
        assert ChoiceMap.choice(Mask(42.0, False)).static_is_empty()

        # Masks with concrete `True` flags have their masks stripped off
        assert ChoiceMap.choice(Mask(42.0, True)) == ChoiceMap.choice(42.0)

        # non-concrete values survive.
        masked_v = Mask(42.0, jnp.array(False))
        assert ChoiceMap.choice(masked_v).get_value() == masked_v

        empty_array = jnp.ones((0,))
        assert ChoiceMap.choice(empty_array).static_is_empty()

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

    def test_extend_through_at(self):
        # Create an initial ChoiceMap
        initial_chm = ChoiceMap.kw(x=1, y={"z": 2})

        # Extend the ChoiceMap using 'at'
        extended_chm = initial_chm.at["y", "w"].set(3)

        # Test that the original values are preserved
        assert extended_chm["x"] == 1
        assert extended_chm["y", "z"] == 2

        # Test that the new value is correctly set
        assert extended_chm["y", "w"] == 3

        # Test that we can chain multiple extensions
        multi_extended_chm = initial_chm.at["y", "w"].set(3).at["a", "b", "c"].set(4)

        assert multi_extended_chm["x"] == 1
        assert multi_extended_chm["y", "z"] == 2
        assert multi_extended_chm["y", "w"] == 3
        assert multi_extended_chm["a", "b", "c"] == 4

        # Test overwriting an existing value
        overwritten_chm = initial_chm.at["y", "z"].set(5)

        assert overwritten_chm["x"] == 1
        assert overwritten_chm["y", "z"] == 5  # Value has been overwritten

        # Test extending with a nested ChoiceMap
        nested_extension = initial_chm.at["nested"].set(ChoiceMap.kw(a=6, b=7))

        assert nested_extension["x"] == 1
        assert nested_extension["y", "z"] == 2
        assert nested_extension["nested", "a"] == 6
        assert nested_extension["nested", "b"] == 7

    def test_filter(self):
        chm = ChoiceMap.kw(x=1, y=2, z=3)
        sel = S["x"] | S["y"]
        filtered = sel.filter(chm)
        assert filtered["x"] == 1
        assert filtered["y"] == 2
        assert "z" not in filtered

    def test_mask(self):
        chm = ChoiceMap.kw(x=1, y=2)
        masked_true = chm.mask(True)
        assert masked_true == chm
        masked_false = chm.mask(False)
        assert masked_false.static_is_empty()

    def test_extend(self):
        chm = ChoiceMap.choice(1)
        extended = chm.extend("a", "b")
        assert extended["a", "b"] == 1

        assert extended.get_value() is None
        assert extended.get_submap("a", "b").get_value() == 1
        assert ChoiceMap.empty().extend("a", "b").static_is_empty()

    def test_switch_chm(self):
        # Test with concrete int index
        chm1 = ChoiceMap.kw(x=1, y=2)
        chm2 = ChoiceMap.kw(a=3, b=4)
        chm3 = ChoiceMap.kw(p=5, q=6)

        switched = ChoiceMap.switch(1, [chm1, chm2, chm3])
        assert switched == chm2

        # Test with array index
        idx = jnp.array(1)
        switched_array = ChoiceMap.switch(idx, [chm1, chm2, chm3])

        # Can get values from any component, masked to the correct idx
        assert switched_array["x"] == Mask(1, jnp.array(False))
        assert switched_array["a"] == Mask(3, jnp.array(True))
        assert switched_array["p"] == Mask(5, jnp.array(False))

        # any statically missing address still raises:
        with pytest.raises(ChoiceMapNoValueAtAddress):
            switched_array["z"]

    def test_or_with_switch(self):
        # Create a switch and a static choice map
        chm1 = ChoiceMap.kw(x=1, y=2)
        chm2 = ChoiceMap.kw(x=3, y=4)
        switch_chm = ChoiceMap.switch(jnp.array(1), [chm1, chm2])
        static_chm = ChoiceMap.kw(z=5)

        # Test Or with switch on left
        or_chm = switch_chm | static_chm

        # Should be a Switch with the static Or'd into each branch
        assert isinstance(or_chm, Switch)
        assert len(or_chm.chms) == 2

        # First branch should have original values masked false
        assert or_chm.chms[0]["x"] == Mask(1, jnp.array(False))
        assert or_chm.chms[0]["y"] == Mask(2, jnp.array(False))
        assert or_chm.chms[0]["z"] == Mask(5, jnp.array(False))

        # Second branch should have original values masked true
        assert or_chm.chms[1]["x"] == Mask(3, jnp.array(True))
        assert or_chm.chms[1]["y"] == Mask(4, jnp.array(True))
        assert or_chm.chms[1]["z"] == Mask(5, jnp.array(True))

        # Test Or with switch on right
        or_chm_2 = static_chm | switch_chm

        # Should be a Switch with the static Or'd into each branch
        assert isinstance(or_chm_2, Switch)
        assert len(or_chm_2.chms) == 2

        # First branch should have original values masked false
        assert or_chm_2.chms[0]["x"] == Mask(1, jnp.array(False))
        assert or_chm_2.chms[0]["y"] == Mask(2, jnp.array(False))
        assert or_chm_2.chms[0]["z"] == Mask(5, jnp.array(False))

        # Second branch should have original values masked true
        assert or_chm_2.chms[1]["x"] == Mask(3, jnp.array(True))
        assert or_chm_2.chms[1]["y"] == Mask(4, jnp.array(True))
        assert or_chm_2.chms[1]["z"] == Mask(5, jnp.array(True))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_or_xor_access(self):
        # Create two choice maps with disjoint addresses
        left = ChoiceMap.kw(x=1, y=2)
        right = ChoiceMap.kw(z=3, w=4)

        # Test Or
        or_chm = left | right
        assert or_chm["x"] == 1  # Only in left
        assert or_chm["y"] == 2  # Only in left
        assert or_chm["z"] == 3  # Only in right
        assert or_chm["w"] == 4  # Only in right

        # Test Xor
        xor_chm = left ^ right
        assert xor_chm["x"] == 1  # Only in left
        assert xor_chm["y"] == 2  # Only in left
        assert xor_chm["z"] == 3  # Only in right
        assert xor_chm["w"] == 4  # Only in right

        # Test that non-existent addresses still raise
        with pytest.raises(ChoiceMapNoValueAtAddress):
            or_chm["does_not_exist"]

        with pytest.raises(ChoiceMapNoValueAtAddress):
            xor_chm["does_not_exist"]

    def test_nested_static_choicemap(self):
        # Create a nested static ChoiceMap
        inner_chm = ChoiceMap.kw(a=1, b=2)
        outer_chm = ChoiceMap.kw(x=inner_chm, y=3)

        # Check that the outer ChoiceMap is a Static
        assert isinstance(outer_chm, Static)

        # Check that the mapping contains the expected structure
        assert len(outer_chm.mapping) == 2
        assert "x" in outer_chm.mapping
        assert "y" in outer_chm.mapping

        # Check that the nested ChoiceMap is stored as a dict in the mapping
        assert isinstance(outer_chm.mapping["x"], dict)
        assert outer_chm.mapping["x"] == {
            "a": ChoiceMap.choice(1),
            "b": ChoiceMap.choice(2),
        }

        # dict is converted back to a Static on the way out.
        assert isinstance(outer_chm.get_submap("x"), Static)

        # Verify values can be accessed correctly
        assert outer_chm["x", "a"] == 1
        assert outer_chm["x", "b"] == 2
        assert outer_chm["y"] == 3

        # Test with a deeper nesting
        deepest_chm = ChoiceMap.kw(m=4, n=5)
        deep_chm = ChoiceMap.kw(p=deepest_chm, q=6)
        root_chm = ChoiceMap.kw(r=deep_chm, s=7)

        # Verify the structure and values
        assert isinstance(root_chm, Static)
        assert isinstance(root_chm.mapping["r"], dict)
        assert isinstance(root_chm.mapping["r"]["p"], dict)
        assert root_chm["r", "p", "m"] == 4
        assert root_chm["r", "p", "n"] == 5
        assert root_chm["r", "q"] == 6
        assert root_chm["s"] == 7

    def test_static_extend(self):
        chm = Static.build({"v": ChoiceMap.choice(1.0), "K": ChoiceMap.empty()})
        assert len(chm.mapping) == 1, "make sure empty chm doesn't make it through"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_simplify(self):
        chm = ChoiceMap.choice(jnp.asarray([2.3, 4.4, 3.3]))
        extended = chm.extend(slice(None, None, None))
        assert extended.simplify() == extended, "no-op with no filters"

        filtered = C["x", "y"].set(2.0).mask(jnp.array(True))
        maskv = Mask(2.0, jnp.array(True))
        assert filtered.simplify() == C["x", "y"].set(maskv), "simplify removes filters"

        xyz = ChoiceMap.d({"x": 1, "y": 2, "z": 3})
        or_chm = xyz.filter(S["x"]) | xyz.filter(S["y"])
        xor_chm = xyz.filter(S["x"]) ^ xyz.filter(S["y"])

        assert or_chm.simplify() == xor_chm.simplify(), "filters pushed down"

        assert or_chm["x"] == 1
        assert or_chm["y"] == 2
        with pytest.raises(ChoiceMapNoValueAtAddress, match="z"):
            or_chm["z"]

        assert or_chm.simplify() == ChoiceMap.d({
            "x": 1,
            "y": 2,
        }), "filters pushed down"

        assert C["x"].set(None).simplify() == C["x"].set(None), "None is not filtered"

    def test_lookup_dynamic(self):
        chm = ChoiceMap.choice(jnp.asarray([2.3, 4.4, 3.3]))
        assert chm.get_submap("x").static_is_empty()
        assert chm[0] == 2.3
        assert chm[1] == 4.4
        assert chm[2] == 3.3

        assert ChoiceMap.empty().extend(slice(None, None, None)).static_is_empty()

    def test_access_dynamic(self):
        # out of order input arrays
        chm = C[jnp.array([4, 8, 2]), "x"].set(jnp.array([4.0, 8.0, 2.0]))
        assert chm[2, "x"] == genjax.Mask(2.0, True)
        assert chm[4, "x"] == genjax.Mask(4.0, True)
        assert chm[8, "x"] == genjax.Mask(8.0, True)

        # indices that don't exist are flagged False.
        assert jnp.array_equal(chm[0, "x"].primal_flag(), jnp.asarray(False))
        assert jnp.array_equal(chm[11, "x"].primal_flag(), jnp.asarray(False))

    def test_merge(self):
        chm1 = ChoiceMap.kw(x=1)
        chm2 = ChoiceMap.kw(y=2)
        merged = chm1.merge(chm2)
        assert merged["x"] == 1
        assert merged["y"] == 2

        # merged is equivalent to or
        assert merged == chm1 | chm2

    def test_get_selection(self):
        chm = ChoiceMap.kw(x=1, y=2)
        sel = chm.get_selection()
        assert sel["x"]
        assert sel["y"]
        assert not sel["z"]

    def test_static_is_empty(self):
        assert ChoiceMap.empty().static_is_empty()
        assert not ChoiceMap.kw(x=1).static_is_empty()

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_xor(self):
        chm1 = ChoiceMap.kw(x=1)
        chm2 = ChoiceMap.kw(y=2)
        xor_chm = chm1 ^ chm2
        assert xor_chm["x"] == 1
        assert xor_chm["y"] == 2

        # Optimization: XorChm.build should return EmptyChm for empty inputs
        assert (ChoiceMap.empty() ^ ChoiceMap.empty()).static_is_empty()

        assert (chm1 ^ ChoiceMap.empty()) == chm1
        assert (ChoiceMap.empty() ^ chm1) == chm1

    def test_or(self):
        chm1 = ChoiceMap.kw(x=1)
        chm2 = ChoiceMap.kw(y=2)
        or_chm = chm1 | chm2
        assert or_chm.get_value() is None
        assert or_chm["x"] == 1
        assert or_chm["y"] == 2

        # Optimization: OrChm.build should return input for empty other input
        assert (chm1 | ChoiceMap.empty()) == chm1
        assert (chm1 | ChoiceMap.empty()) == chm1
        assert (ChoiceMap.empty() | chm1) == chm1

        x_masked = ChoiceMap.choice(2.0).mask(jnp.asarray(True))
        y_masked = ChoiceMap.choice(3.0).mask(jnp.asarray(True))
        assert (x_masked | y_masked).get_value().unmask() == 2.0

        with pytest.raises(Exception, match="Choice and non-Choice in Or"):
            _ = C["x"].set(1.0) | C["x", "y"].set(2.0)

    def test_and(self):
        chm1 = ChoiceMap.kw(x=1, y=2, z=3)
        chm2 = ChoiceMap.kw(y=20, z=30, w=40)

        and_chm = chm1 & chm2

        # Check that only common keys are present
        assert "x" not in and_chm
        assert "y" in and_chm
        assert "z" in and_chm
        assert "w" not in and_chm

        # Check that values come from the right-hand side (chm2)
        assert and_chm["y"] == 20
        assert and_chm["z"] == 30

        # Test with empty ChoiceMap
        empty_chm = ChoiceMap.empty()
        assert (chm1 & empty_chm).static_is_empty()
        assert (empty_chm & chm1).static_is_empty()

        # Test with nested ChoiceMaps
        nested_chm1 = ChoiceMap.kw(a={"b": 1, "c": 2}, d=3)
        nested_chm2 = ChoiceMap.kw(a={"b": 10, "d": 20}, d=30)
        nested_and_chm = nested_chm1 & nested_chm2

        assert nested_and_chm["a", "b"] == 10
        assert "c" not in nested_and_chm("a")
        assert "d" not in nested_and_chm("a")
        assert nested_and_chm["d"] == 30

    def test_call(self):
        chm = ChoiceMap.kw(x={"y": 1})
        assert chm("x")("y") == ChoiceMap.choice(1)

    def test_getitem(self):
        chm = ChoiceMap.kw(x=1)
        assert chm["x"] == 1
        with pytest.raises(ChoiceMapNoValueAtAddress, match="y"):
            chm["y"]

    def test_contains(self):
        chm = ChoiceMap.kw(x={"y": 1})
        assert "x" not in chm
        assert "y" in chm("x")
        assert ("x", "y") in chm
        assert "z" not in chm

    def test_choicemap_filter_with_wildcard(self):
        xs = jnp.array([1.0, 2.0, 3.0])
        ys = jnp.array([4.0, 5.0, 6.0])
        # Create a ChoiceMap with values at 'x' and 'y' addresses
        chm = C[:].set({"x": xs, "y": ys})

        # Create a Selection for 'x'
        sel = S["x"]

        # Filter the ChoiceMap using the Selection
        filtered_chm = chm.filter(sel)

        # Assert that only 'x' values are present in the filtered ChoiceMap
        assert jnp.all(filtered_chm[:, "x"] == jnp.array([1.0, 2.0, 3.0]))

        # Assert that 'y' values are not present in the filtered ChoiceMap
        with pytest.raises(ChoiceMapNoValueAtAddress):
            filtered_chm[:, "y"]

        # Assert that the structure of the filtered ChoiceMap is preserved
        assert filtered_chm[0, "x"] == 1.0
        assert filtered_chm[1, "x"] == 2.0
        assert filtered_chm[2, "x"] == 3.0

    def test_filtered_chm_update(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(10.0, 1.0) @ "y"
            return x, y

        key = jax.random.key(0)
        tr = f.repeat(n=4).simulate(key, ())

        xs = jnp.ones(4)
        ys = 5 * jnp.ones(4)
        constraint = C[:].set({"x": xs, "y": ys})
        only_xs = constraint.filter(S["x"])
        only_ys = constraint.filter(S["y"])

        key, subkey = jax.random.split(key)
        new_tr, _, _, _ = tr.update(subkey, only_xs)
        new_choices = new_tr.get_choices()
        assert jnp.array_equal(new_choices[:, "x"], xs)
        assert not jnp.array_equal(new_choices[:, "y"], ys)

        key, subkey = jax.random.split(key)
        new_tr_2, _, _, _ = tr.update(subkey, only_ys)
        new_choices_2 = new_tr_2.get_choices()
        assert not jnp.array_equal(new_choices_2[:, "x"], xs)
        assert jnp.array_equal(new_choices_2[:, "y"], ys)

    def test_choicemap_with_static_idx(self):
        chm = C[0].set({"x": 1.0, "y": 2.0})

        # if the index is NOT an array (i.e. statically known) we get a static value out, not a mask.
        assert chm[0, "x"] == 1.0
        assert chm[0, "y"] == 2.0

    def test_chm_roundtrip(self):
        chm = ChoiceMap.choice(3.0)
        assert chm == chm.__class__.from_attributes(**chm.attributes_dict())

    def test_choicemap_validation(self):
        @genjax.gen
        def model(x):
            y = genjax.normal(x, 1.0) @ "y"
            z = genjax.bernoulli(probs=0.5) @ "z"
            return y + z

        # Valid ChoiceMap
        valid_chm = ChoiceMap.kw(y=1.0, z=1)
        assert valid_chm.invalid_subset(model, (0.0,)) is None

        # Invalid ChoiceMap - missing 'z'
        invalid_chm1 = ChoiceMap.kw(x=1.0)
        assert invalid_chm1.invalid_subset(model, (0.0,)) == invalid_chm1

        # Invalid ChoiceMap - extra address
        invalid_chm2 = ChoiceMap.kw(y=1.0, z=1, extra=0.5)
        assert invalid_chm2.invalid_subset(model, (0.0,)) == ChoiceMap.kw(extra=0.5)

    def test_choicemap_nested_validation(self):
        @genjax.gen
        def inner_model():
            a = genjax.normal(0.0, 1.0) @ "a"
            b = genjax.bernoulli(probs=0.5) @ "b"
            return a + b

        @genjax.gen
        def outer_model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = inner_model() @ "y"
            return x + y

        # Valid nested ChoiceMap
        valid_nested_chm = ChoiceMap.kw(x=1.0, y=ChoiceMap.kw(a=0.5, b=1))
        assert valid_nested_chm.invalid_subset(outer_model, ()) is None

        # Invalid nested ChoiceMap - missing inner 'b'
        invalid_nested_chm1 = ChoiceMap.kw(x=1.0, y=ChoiceMap.kw(a=0.5))
        assert invalid_nested_chm1.invalid_subset(outer_model, ()) is None, (
            "missing address is fine"
        )

        # Invalid nested ChoiceMap - extra address in inner model
        invalid_nested_chm2 = ChoiceMap.kw(x=1.0, y=ChoiceMap.kw(a=0.5, b=1, c=2.0))
        assert invalid_nested_chm2.invalid_subset(outer_model, ()) == ChoiceMap.kw(
            y=ChoiceMap.kw(c=2.0)
        )

        # Invalid nested ChoiceMap - extra address in outer model
        invalid_nested_chm3 = ChoiceMap.kw(x=1.0, y=ChoiceMap.kw(a=0.5, b=1), z=3.0)
        assert invalid_nested_chm3.invalid_subset(outer_model, ()) == ChoiceMap.kw(
            z=3.0
        )

    def test_choicemap_nested_vmap(self):
        @genjax.gen
        def inner_model(x):
            a = genjax.normal(x, 1.0) @ "a"
            b = genjax.bernoulli(probs=0.5) @ "b"
            return a + b

        @genjax.gen
        def outer_model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = inner_model.vmap(in_axes=(0,))(jnp.array([1.0, 2.0, 3.0])) @ "y"
            return x + jnp.sum(y)

        # Valid nested ChoiceMap with vmap
        valid_vmap_chm = ChoiceMap.kw(
            x=1.0,
            y=C[:].set(
                ChoiceMap.kw(a=jnp.array([0.5, 1.5, 2.5]), b=jnp.array([1, 0, 1]))
            ),
        )
        assert valid_vmap_chm.invalid_subset(outer_model, ()) is None

        # Invalid nested ChoiceMap - wrong shape for vmapped inner model
        inner_chm = ChoiceMap.kw(a=jnp.array([0.5, 1.5, 2.5]), b=jnp.array([1, 0, 1]))
        invalid_vmap_chm1 = ChoiceMap.kw(
            x=1.0,
            # missing the index nesting is fine, we don't care anymore
            y=inner_chm,
        )
        assert invalid_vmap_chm1.invalid_subset(outer_model, ()) is None

        # Invalid nested ChoiceMap - extra address in vmapped inner model

        invalid_vmap_chm2 = ChoiceMap.kw(
            x=1.0,
            y=C[:].set(
                ChoiceMap.kw(
                    a=jnp.array([0.5, 1.5, 2.5]),
                    b=jnp.array([1, 0, 1]),
                    c=jnp.array([0.1, 0.2, 0.3]),  # Extra address
                )
            ),
        )
        expected_result = C["y", :, "c"].set(jnp.array([0.1, 0.2, 0.3]))
        actual_result = invalid_vmap_chm2.invalid_subset(outer_model, ())
        assert jtu.tree_structure(actual_result) == jtu.tree_structure(expected_result)
        assert jtu.tree_all(
            jtu.tree_map(
                lambda x, y: jnp.allclose(x, y), actual_result, expected_result
            )
        )

    def test_choicemap_switch(self):
        @genjax.gen
        def model1():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.gen
        def model2():
            y = genjax.uniform(0.0, 1.0) @ "y"
            return y

        @genjax.gen
        def model3():
            z = genjax.normal(0.0, 1.0) @ "z"
            return z

        switch_model = genjax.switch(model1, model2, model3)

        @genjax.gen
        def outer_model():
            choice = genjax.categorical(probs=[0.3, 0.3, 0.4]) @ "choice"
            return switch_model(choice, (), (), ()) @ "out"

        # Valid ChoiceMap for model1
        valid_chm1 = ChoiceMap.kw(choice=0, out={"x": 0.5})
        assert valid_chm1.invalid_subset(outer_model, ()) is None

        # Valid ChoiceMap for model2
        valid_chm2 = ChoiceMap.kw(choice=1, out={"y": 0.7})
        assert valid_chm2.invalid_subset(outer_model, ()) is None

        # Valid ChoiceMap for model3
        valid_chm3 = ChoiceMap.kw(choice=2, out={"z": 1.2})
        assert valid_chm3.invalid_subset(outer_model, ()) is None

        # Valid ChoiceMap with entries for all models
        valid_chm_all = ChoiceMap.kw(choice=0, out={"x": 0.5, "y": 0.7, "z": 1.2})
        assert valid_chm_all.invalid_subset(outer_model, ()) is None

        # Invalid ChoiceMap - extra address
        invalid_chm2 = ChoiceMap.kw(choice=1, out={"q": 0.5})
        assert invalid_chm2.invalid_subset(outer_model, ()) == C["out", "q"].set(0.5)
        pass

    def test_choicemap_scan(self):
        @genjax.gen
        def inner_model(mean):
            return genjax.normal(mean, 1.0) @ "x"

        outer_model = inner_model.iterate(n=4)

        # Test valid ChoiceMap
        valid_chm = C[:, "x"].set(jnp.array([0.5, 1.2, 0.8, 0.9]))
        assert valid_chm.invalid_subset(outer_model, (1.0,)) is None

        # index layer isn't required, forgetting it is fine
        invalid_chm2 = C["x"].set(jnp.array([0.5, 1.2, 0.8, 0.9]))
        assert invalid_chm2.invalid_subset(outer_model, (1.0,)) is None

        xs = jnp.array([0.5, 1.2, 0.8, 0.9])
        zs = jnp.array([0.5, 1.2, 0.8, 0.9])
        invalid_chm3 = C[:].set({"x": xs, "z": zs})
        invalid_subset = invalid_chm3.invalid_subset(outer_model, (1.0,))
        expected_invalid = C[:, "z"].set(zs)
        assert jtu.tree_structure(invalid_subset) == jtu.tree_structure(
            expected_invalid
        )
        assert jtu.tree_all(
            jtu.tree_map(
                lambda x, y: jnp.allclose(x, y), invalid_subset, expected_invalid
            )
        )

    def test_choicemap_slice(self):
        # partial slices are not allowed when setting:
        with pytest.raises(ValueError):
            C[:3, "x"].set(jnp.array([1, 2]))

        with pytest.raises(ValueError):
            C[0:3, "x"].set(jnp.array([1, 2]))

        with pytest.raises(ValueError):
            C[0:3:1, "x"].set(jnp.array([1, 2]))

        # set with a full slice
        vals = jnp.arange(10)
        chm = C[:, "x"].set(vals)

        # Full lookup:
        assert jnp.array_equal(chm[:, "x"], jnp.arange(10))

        # single int index:
        assert chm[1, "x"] == vals[1]

        # single array index:
        assert chm[jnp.array(5), "x"] == vals[5]

        # one non-full slices is allowed:
        assert jnp.array_equal(chm[0:4, "x"], vals[0:4])

        assert jnp.array_equal(chm[0:4, "x"], vals[0:4])

    def test_nested_masking(self):
        chm = C[jnp.array(0), "w", jnp.array(1), :, :].set(jnp.ones((3, 2, 2)))
        assert jnp.array_equal(chm[0, "w", 1, :, :].unmask(), jnp.ones((3, 2, 2)))

    def test_choicemap_slice_validation(self):
        # Creation with scalar and string keys
        chm = C[0, "x", 1].set(10)
        assert chm[0, "x", 1] == 10

        # Creation with IntArray (shape == ())
        idx = jnp.array(2, dtype=jnp.int32)
        chm = C[idx, "y"].set(20)
        assert chm[2, "y"] == genjax.Mask(20, True)

        # Creation with optional single array of indices
        indices = jnp.array([0, 1, 2])
        values = jnp.array([5, 10, 15])
        chm = C["z", indices].set(values)

        # querying array-shaped indices with a slice is not allowed:
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Slices are not allowed against array-shaped dynamic addresses."
            ),
        ):
            chm["z", :]

        # Creation with full slices after array
        chm = C[0, "w", indices, :, :].set(jnp.ones((3, 2, 2)))
        assert jnp.array_equal(chm[0, "w", 1, :, :].unmask(), jnp.ones((2, 2)))

        # Lookup with scalar and string keys
        assert chm[0, "w", 1, 0, 0] == genjax.Mask(1, True)

        # Lookup with IntArray (shape == ())
        assert chm[0, "w", idx, 0, 0] == genjax.Mask(1, True)

        # Lookup with single partial slice
        partial_result = chm[0, "w", 0, 1:3, :]
        assert jnp.array_equal(partial_result.unmask(), jnp.ones((1, 2)))
        assert jnp.array_equal(partial_result.primal_flag(), jnp.array(True))

        # Lookup with full slices
        full_result = jax.vmap(lambda i: chm[0, "w", i, :, :])(indices)
        assert full_result.unmask().shape == (3, 2, 2)

        # Ensure dynamic components are deferred to leaf
        complex_chm = C[0, "a", indices, :, "b"].set(jnp.ones((3, 2)))
        assert jnp.array_equal(complex_chm[0, "a", 1, :, "b"].unmask(), jnp.ones(2))

        # Verify that partial slices are not allowed in creation
        with pytest.raises(ValueError):
            C[0, "x", 1:3].set(jnp.array([1, 2]))

        # Verify that multiple arrays are not allowed in creation
        with pytest.raises(ValueError):
            C[indices, indices].set(jnp.ones((3, 3)))

        # Verify that partial slices are allowed in lookup
        assert complex_chm[0, "a", 0, 1:3, "b"] == genjax.Mask(jnp.array([1.0]), True)


dictionaries_for_choice_maps = st.deferred(
    lambda: st.dictionaries(
        st.text(),
        st.floats(allow_nan=False)
        | st.lists(st.floats(allow_nan=False))
        | dictionaries_for_choice_maps,
        min_size=1,
    )
)


def all_paths(mapping) -> list[tuple[tuple[StaticAddress, ...], Any]]:
    paths = []
    stack: list[tuple[StaticAddress, Any]] = [((), mapping)]
    while stack:
        prefix, mapping = stack.pop()
        if isinstance(mapping, dict) and mapping:
            for k, v in mapping.items():
                stack.append(((*prefix, k), v))
        else:
            paths.append((prefix, mapping))
    return paths


class TestSubmap:
    @given(dictionaries_for_choice_maps, st.data())
    def test_get_submap_split_path(self, mapping, data):
        choice_map = ChoiceMap.d(mapping)
        paths = all_paths(mapping)

        path, value = data.draw(st.sampled_from(paths))

        assume(path)

        i = data.draw(st.integers(0, len(path)))

        assert choice_map.get_submap(path[:i])[path[i:]] == value, (
            "a path split between get_submap and [] will reach the value"
        )
        assert choice_map.get_submap(path[:i], path[i:]) == choice_map.get_submap(
            path
        ), (
            "get_submap can take multiple path-segments and reach the same leaf as a full path"
        )

    @given(dictionaries_for_choice_maps, st.data())
    def test_path_can_be_splat(self, mapping, data):
        choice_map = ChoiceMap.d(mapping)
        paths = all_paths(mapping)

        path, _ = data.draw(st.sampled_from(paths))

        assume(path)

        assert choice_map.get_submap(path) == choice_map.get_submap(*path), (
            "Splatting out a path returns the same result as providing the tuple of path segments"
        )
