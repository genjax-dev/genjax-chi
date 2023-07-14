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

import genjax
import jax.tree_util as jtu


class TestHashableDict:
    def test_sorting(self):
        # Test construction.
        h = genjax.core.HashableDict({1 : 5, "1" : 10})
        values, tree_def = jtu.tree_flatten(h)
        new_h = jtu.tree_unflatten(tree_def, values)
        assert h == new_h
