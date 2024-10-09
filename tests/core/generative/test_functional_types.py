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
