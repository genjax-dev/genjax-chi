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


from genjax._src.core.interpreters.incremental import Diff


class TestDiff:
    def test_no_nested_diffs(self):
        d1 = Diff.no_change(1.0)
        d2 = Diff.unknown_change(d1)
        assert not isinstance(d2.get_primal(), Diff)

        assert Diff.static_check_no_change(d1)
        assert not Diff.static_check_no_change(d2)
