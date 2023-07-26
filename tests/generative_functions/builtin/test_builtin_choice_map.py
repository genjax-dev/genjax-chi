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


class TestBuiltinChoiceMap:
    def test_builtin_choice_map(self):
        chm = genjax.choice_map({})
        chm["x"] = 0.5
        chm["y"] = 0.3
        chm["z", "x"] = 0.2
        assert chm["x"] == 0.5
        assert chm["y"] == 0.3
        assert chm["z", "x"] == 0.2
