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

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from genjax._src.checkify import do_checkify
from genjax._src.core.generative.functional_types import Mask


class TestMask:
    def test_mask_unmask_without_default(self):
        valid_mask = Mask(42, True)
        assert valid_mask.unmask() == 42

        invalid_mask = Mask(42, False)
        with do_checkify():
            with pytest.raises(Exception):
                invalid_mask.unmask()

    def test_mask_unmask_with_default(self):
        valid_mask = Mask(42, True)
        assert valid_mask.unmask(default=0) == 42

        invalid_mask = Mask(42, False)
        assert invalid_mask.unmask(default=0) == 0

    def test_mask_unmask_pytree(self):
        pytree = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        valid_mask = Mask(pytree, True)
        assert valid_mask.unmask() == pytree

        invalid_mask = Mask(pytree, False)
        default = {"a": 0, "b": [0, 0], "c": {"d": 0}}
        result = invalid_mask.unmask(default=default)
        assert result == default

    def test_build(self):
        mask = Mask.build(42, True)
        assert isinstance(mask, Mask)
        assert mask.flag is True
        assert mask.value == 42

        nested_mask = Mask.build(Mask.build(42, True), False)
        assert isinstance(nested_mask, Mask)
        assert nested_mask.flag is False
        assert nested_mask.value == 42

    def test_build_flag_validation(self):
        # Boolean flags should be left unchanged
        mask = Mask.build(42, True)
        assert mask.flag is True

        mask = Mask.build([1, 2, 3], False)
        assert mask.flag is False

        # Array flags should only be allowed if they can be broadcast to match value shape
        value = jnp.array([1.0, 2.0, 3.0])
        flag = jnp.array([True])
        mask = Mask.build(value, flag)
        assert jnp.array_equal(mask.primal_flag(), jnp.array([True]))

        # Works with pytrees
        value = {"a": jnp.ones((3, 2)), "b": jnp.ones((3, 2))}
        flag = jnp.array([True, False])
        mask = Mask.build(value, flag)
        assert jnp.array_equal(mask.primal_flag(), flag)

        # differing shapes in pytree leaves
        with pytest.raises(
            ValueError, match="All leaves in value must have same shape"
        ):
            value = {"a": jnp.ones((4, 8)), "b": jnp.ones((3, 2))}
            flag = jnp.array([True, False])
            mask = Mask.build(value, flag)

        # Incompatible shapes should raise error
        value = jnp.array([1.0, 2.0])
        flag = jnp.array([True, False, True])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Flag [ True False  True] cannot be broadcast to shape (2,)"
            ),
        ):
            Mask.build(value, flag)

    def test_maybe_mask(self):
        result = Mask.maybe_mask(42, True)
        assert result == 42

        result = Mask.maybe_mask(42, False)
        assert result is None

        mask = Mask(42, True)
        assert Mask.maybe_mask(mask, True) == 42
        assert Mask.maybe_mask(mask, False) is None

        assert Mask.maybe_mask(None, jnp.asarray(True)) == Mask(
            None, jnp.asarray(True)
        ), "None survives maybe_mask"

    def test_mask_or_concrete_flags(self):
        # True | True = True
        mask1 = Mask(42, True)
        mask2 = Mask(43, True)
        result = mask1 | mask2
        assert result.primal_flag() is True
        assert result.value == 42

        # True | False = True (takes first value)
        mask1 = Mask(42, True)
        mask2 = Mask(43, False)
        result = mask1 | mask2
        assert result.primal_flag() is True
        assert result.value == 42

        # False | True = True (takes second value)
        mask1 = Mask(42, False)
        mask2 = Mask(43, True)
        result = mask1 | mask2
        assert result.primal_flag() is True
        assert result.value == 43

        # False | False = False
        mask1 = Mask(42, False)
        mask2 = Mask(43, False)
        result = mask1 | mask2
        assert result.primal_flag() is False

        # Array flags result in array flag
        mask1 = Mask(jnp.array([42, 42, 42, 42]), jnp.array([True, True, False, False]))
        mask2 = Mask(jnp.array([43, 43, 43, 43]), jnp.array([False, True, False, True]))
        result = mask1 | mask2
        jtu.tree_map(
            jnp.array_equal,
            result,
            Mask(jnp.array([42, 43, 43, 42]), jnp.array([True, True, False, True])),
        )

    def test_mask_xor_concrete_flags(self):
        # True ^ True = False
        mask1 = Mask(42, True)
        mask2 = Mask(43, True)
        result = mask1 ^ mask2
        assert result.primal_flag() is False

        # True ^ False = True (takes first value)
        mask1 = Mask(42, True)
        mask2 = Mask(43, False)
        result = mask1 ^ mask2
        assert result.primal_flag() is True
        assert result.value == 42

        # False ^ True = True (takes second value)
        mask1 = Mask(42, False)
        mask2 = Mask(43, True)
        result = mask1 ^ mask2
        assert result.primal_flag() is True
        assert result.value == 43

        # False ^ False = False
        mask1 = Mask(42, False)
        mask2 = Mask(43, False)
        result = mask1 ^ mask2
        assert result.primal_flag() is False

        # Array flags result in array flag
        mask1 = Mask(jnp.array([42, 42, 42, 42]), jnp.array([True, True, False, False]))
        mask2 = Mask(jnp.array([43, 43, 43, 43]), jnp.array([False, True, False, True]))
        result = mask1 ^ mask2
        jtu.tree_map(
            jnp.array_equal,
            result,
            Mask(jnp.array([42, 42, 43, 42]), jnp.array([True, False, False, True])),
        )

    def test_mask_combine_different_pytree_shapes(self):
        mask1 = Mask({"a": 1, "b": 2}, True)
        mask2 = Mask({"a": 1}, True)

        with pytest.raises(
            ValueError, match="Cannot combine masks with different tree structures"
        ):
            _ = mask1 | mask2

        with pytest.raises(
            ValueError, match="Cannot combine masks with different tree structures"
        ):
            _ = mask1 ^ mask2

    def test_mask_combine_different_array_shapes(self):
        # Array vs array with different shapes
        mask1 = Mask(jnp.ones((2, 3)), True)
        mask2 = Mask(jnp.ones((2, 2)), True)

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask1 | mask2

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask1 ^ mask2

        # Scalar vs array
        mask3 = Mask(jnp.asarray(1.0), True)
        mask4 = Mask(jnp.ones((2, 2)), True)

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask3 | mask4

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask3 ^ mask4

        # Different scalar shapes
        mask5 = Mask(1.0, True)
        mask6 = Mask(jnp.array(1.0), True)

        assert mask5 | mask6 == mask6  # pyright: ignore

        assert (mask5 ^ mask6).primal_flag() is False  # pyright: ignore

        # Same scalar shapes should work
        mask7 = Mask(1.0, True)
        mask8 = Mask(2.0, False)
        assert mask7 | mask8 == mask7
        assert mask7 ^ mask8 == mask7
