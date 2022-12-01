# Copyright 2022 MIT Probabilistic Computing Project
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
import pytest

import genjax


key = jax.random.PRNGKey(314159)


class TestBuiltinSelection:
    def test_builtin_selection(self):
        new = genjax.BuiltinSelection.new(["x", ("z", "y")])
        assert new.has_subtree("z")
        assert new.has_subtree("x")
        v = new["x"]
        assert isinstance(v, AllSelection)
        v = new["z", "y"]
        assert isinstance(v, AllSelection)
