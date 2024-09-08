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


from genjax import ChoiceMap
from genjax import ChoiceMapBuilder as C
from genjax import SelectionBuilder as S


class TestChoiceMap:
    def test_value_map(self):
        value_chm = ChoiceMap.value(3.0)
        assert 3.0 == value_chm.get_value()
        assert () in value_chm

    def test_address_map(self):
        chm = C["x"].set(3.0)
        assert chm["x"] == 3.0


class TestSelections:
    def test_selection(self):
        new = S["x"] | S["z", "y"]
        assert new["x"]
        assert new["z", "y"]
        new = S["x"]
        assert new["x"]
        assert new["x", "y"]
        assert new["x", "y", "z"]
        new = S["x", "y", "z"]
        assert new["x", "y", "z"]
        assert not new["x"]
        assert not new["x", "y"]
