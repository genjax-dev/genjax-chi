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

import jax.numpy as jnp
import pytest

from genjax._src.checkify import do_checkify
from genjax._src.core.generative.functional_types import Mask, staged_choose


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

    def test_mask_maybe(self):
        mask = Mask.maybe(42, True)
        assert isinstance(mask, Mask)
        assert mask.flag is True
        assert mask.value == 42

        nested_mask = Mask.maybe(Mask(42, True), False)
        assert isinstance(nested_mask, Mask)
        assert nested_mask.flag is False
        assert nested_mask.value == 42

    def test_mask_maybe_none(self):
        result = Mask.maybe_none(42, True)
        assert result == 42

        result = Mask.maybe_none(42, False)
        assert result is None

        mask = Mask(42, True)
        result = Mask.maybe_none(mask, True)
        assert isinstance(result, Mask)
        assert result.flag is True
        assert result.value == 42


class TestStagedChoose:
    def test_static_integer_index(self):
        result = staged_choose(1, [10, 20, 30])
        assert result == 20

    def test_jax_array_index(self):
        """
        Test that staged_choose works correctly with JAX array indices.
        This test ensures that when given a JAX array as an index,
        the function selects the correct value from the list.
        """
        result = staged_choose(jnp.array(2), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(30))

    def test_heterogeneous_types(self):
        """
        Test that staged_choose correctly handles heterogeneous types.
        It should attempt to cast compatible types (like bool to int)
        and use the dtype of the result for consistency.
        """
        result = staged_choose(2, [True, 2, False])
        assert result == 0
        assert jnp.asarray(result).dtype == jnp.int32

    def test_wrap_mode(self):
        """
        Test that staged_choose wraps around when the index is out of bounds.
        This should work for both jnp.array indices and concrete integer indices.
        """
        # first, the jnp.array index case:
        result = staged_choose(jnp.array(3), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(10))

        # then the concrete index case:
        concrete_result = staged_choose(3, [10, 20, 30])
        assert jnp.array_equal(result, concrete_result)
