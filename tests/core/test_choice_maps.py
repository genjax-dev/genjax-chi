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
import jax.tree_util as jtu
import pytest

import genjax
from genjax import ChoiceMap, Selection
from genjax import ChoiceMapBuilder as C
from genjax import SelectionBuilder as S
from genjax._src.core.generative.choice_map import ChoiceMapNoValueAtAddress, Static
from genjax._src.core.generative.functional_types import Mask


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

    def test_selection_leaf(self):
        leaf_sel = Selection.leaf().extend("x", "y")
        assert not leaf_sel["x"]
        assert leaf_sel["x", "y"]

        # only exact matches are allowed
        assert not leaf_sel["x", "y", "z"]

        # wildcards work
        assert leaf_sel[..., "y"]

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

    def test_selection_mask(self):
        sel = S["x"] | S["y"]
        masked_sel = sel.mask(jnp.asarray(True))
        assert masked_sel["x"]
        assert masked_sel["y"]
        assert not masked_sel["z"]

        masked_sel = sel.mask(False)
        assert not masked_sel["x"]
        assert not masked_sel["y"]
        assert not masked_sel["z"]

        # bool works like flags
        assert sel.mask(True) == sel.mask(True)
        assert sel.mask(False) == sel.mask(False)

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
    def test_empty(self):
        empty_chm = ChoiceMap.empty()
        assert empty_chm.static_is_empty()

    def test_value(self):
        value_chm = ChoiceMap.choice(42.0)
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

        # ... is a wildcard
        assert extended[..., "b"] == 1
        assert extended["a", ...] == 1

        assert extended.get_value() is None
        assert extended.get_submap("a").get_submap("b").get_value() == 1
        assert ChoiceMap.empty().extend("a", "b").static_is_empty()

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

    def test_simplify(self):
        chm = ChoiceMap.choice(jnp.asarray([2.3, 4.4, 3.3]))
        extended = chm.extend(jnp.array([0, 1, 2]))
        assert extended.simplify() == extended, "no-op with no filters"

        filtered = C["x", "y"].set(2.0).mask(jnp.array(True))
        maskv = Mask(2.0, jnp.array(True))
        assert filtered.simplify() == C["x", "y"].set(maskv), "simplify removes filters"

        xyz = ChoiceMap.d({"x": 1, "y": 2, "z": 3})
        or_chm = xyz.filter(S["x"]) | xyz.filter(S["y"].mask(jnp.array(True)))

        xor_chm = xyz.filter(S["x"]) ^ xyz.filter(S["y"].mask(jnp.array(True)))

        assert or_chm.simplify() == xor_chm.simplify(), "filters pushed down"

        assert or_chm["x"] == 1
        assert or_chm["y"] == maskv
        with pytest.raises(ChoiceMapNoValueAtAddress, match="z"):
            or_chm["z"]

        assert or_chm.simplify() == ChoiceMap.d({
            "x": 1,
            "y": maskv,
        }), "filters pushed down"

        assert C["x"].set(None).simplify() == C["x"].set(None), "None is not filtered"

    def test_extend_dynamic(self):
        chm = ChoiceMap.choice(jnp.asarray([2.3, 4.4, 3.3]))
        extended = chm.extend(jnp.array([0, 1, 2]))
        assert extended.get_value() is None
        assert extended.get_submap("x").static_is_empty()
        assert extended[0].unmask() == 2.3
        assert extended[1].unmask() == 4.4
        assert extended[2].unmask() == 3.3

        assert ChoiceMap.empty().extend(jnp.array([0, 1, 2])).static_is_empty()

    def test_merge(self):
        chm1 = ChoiceMap.kw(x=1)
        chm2 = ChoiceMap.kw(y=2)
        merged = chm1.merge(chm2)
        assert merged["x"] == 1
        assert merged["y"] == 2

        # merged is equivalent to xor
        assert merged == chm1 ^ chm2

    def test_get_selection(self):
        chm = ChoiceMap.kw(x=1, y=2)
        sel = chm.get_selection()
        assert sel["x"]
        assert sel["y"]
        assert not sel["z"]

    def test_static_is_empty(self):
        assert ChoiceMap.empty().static_is_empty()
        assert not ChoiceMap.kw(x=1).static_is_empty()

    def test_xor(self):
        chm1 = ChoiceMap.kw(x=1)
        chm2 = ChoiceMap.kw(y=2)
        xor_chm = chm1 ^ chm2
        assert xor_chm["x"] == 1
        assert xor_chm["y"] == 2

        with pytest.raises(
            Exception,
            match="The disjoint union of two choice maps have a value collision",
        ):
            (chm1 ^ chm1)["x"]

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
        chm = C[jnp.arange(3)].set({"x": xs, "y": ys})

        # Create a Selection with a wildcard for 'x'
        sel = S[..., "x"]

        # Filter the ChoiceMap using the Selection
        filtered_chm = chm.filter(sel)

        # Assert that only 'x' values are present in the filtered ChoiceMap
        assert jnp.all(filtered_chm[..., "x"] == jnp.array([1.0, 2.0, 3.0]))

        # Assert that 'y' values are not present in the filtered ChoiceMap
        with pytest.raises(ChoiceMapNoValueAtAddress):
            filtered_chm[..., "y"]

        # Assert that the structure of the filtered ChoiceMap is preserved
        assert filtered_chm[0, "x"].unmask() == 1.0
        assert filtered_chm[1, "x"].unmask() == 2.0
        assert filtered_chm[2, "x"].unmask() == 3.0

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
        constraint = C[jnp.arange(4)].set({"x": xs, "y": ys})
        only_xs = constraint.filter(S[..., "x"])
        only_ys = constraint.filter(S[..., "y"])

        key, subkey = jax.random.split(key)
        new_tr, _, _, _ = tr.update(subkey, only_xs)
        new_choices = new_tr.get_choices()
        assert jnp.array_equal(new_choices[..., "x"], xs)
        assert not jnp.array_equal(new_choices[..., "y"], ys)

        key, subkey = jax.random.split(key)
        new_tr_2, _, _, _ = tr.update(subkey, only_ys)
        new_choices_2 = new_tr_2.get_choices()
        assert not jnp.array_equal(new_choices_2[..., "x"], xs)
        assert jnp.array_equal(new_choices_2[..., "y"], ys)

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
            z = genjax.bernoulli(0.5) @ "z"
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
            b = genjax.bernoulli(0.5) @ "b"
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
        assert (
            invalid_nested_chm1.invalid_subset(outer_model, ()) is None
        ), "missing address is fine"

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
            b = genjax.bernoulli(0.5) @ "b"
            return a + b

        @genjax.gen
        def outer_model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = inner_model.vmap(in_axes=(0,))(jnp.array([1.0, 2.0, 3.0])) @ "y"
            return x + jnp.sum(y)

        # Valid nested ChoiceMap with vmap
        valid_vmap_chm = ChoiceMap.kw(
            x=1.0,
            y=C[jnp.arange(3)].set(
                ChoiceMap.kw(a=jnp.array([0.5, 1.5, 2.5]), b=jnp.array([1, 0, 1]))
            ),
        )
        assert valid_vmap_chm.invalid_subset(outer_model, ()) is None

        # Invalid nested ChoiceMap - wrong shape for vmapped inner model
        inner_chm = ChoiceMap.kw(a=jnp.array([0.5, 1.5, 2.5]), b=jnp.array([1, 0, 1]))
        invalid_vmap_chm1 = ChoiceMap.kw(
            x=1.0,
            # missing the index nesting
            y=inner_chm,
        )
        assert invalid_vmap_chm1.invalid_subset(outer_model, ()) == C["y"].set(
            inner_chm
        )

        # Invalid nested ChoiceMap - extra address in vmapped inner model

        invalid_vmap_chm2 = ChoiceMap.kw(
            x=1.0,
            y=C[jnp.arange(3)].set(
                ChoiceMap.kw(
                    a=jnp.array([0.5, 1.5, 2.5]),
                    b=jnp.array([1, 0, 1]),
                    c=jnp.array([0.1, 0.2, 0.3]),  # Extra address
                )
            ),
        )
        expected_result = C["y", jnp.arange(3), "c"].set(jnp.array([0.1, 0.2, 0.3]))
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
            choice = genjax.categorical([0.3, 0.3, 0.4]) @ "choice"
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
        valid_chm = C[jnp.arange(4), "x"].set(jnp.array([0.5, 1.2, 0.8, 0.9]))
        assert valid_chm.invalid_subset(outer_model, (1.0,)) is None

        # forgot the index layer
        invalid_chm2 = C["x"].set(jnp.array([0.5, 1.2, 0.8, 0.9]))
        assert invalid_chm2.invalid_subset(outer_model, (1.0,)) == invalid_chm2

        xs = jnp.array([0.5, 1.2, 0.8, 0.9])
        zs = jnp.array([0.5, 1.2, 0.8, 0.9])
        invalid_chm3 = C[jnp.arange(4)].set({"x": xs, "z": zs})
        invalid_subset = invalid_chm3.invalid_subset(outer_model, (1.0,))
        expected_invalid = C[jnp.arange(4), "z"].set(zs)
        assert jtu.tree_structure(invalid_subset) == jtu.tree_structure(
            expected_invalid
        )
        assert jtu.tree_all(
            jtu.tree_map(
                lambda x, y: jnp.allclose(x, y), invalid_subset, expected_invalid
            )
        )
