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


from genjax._src.core.interpreters.incremental import Diff, NoChange, UnknownChange


class TestDiff:
    def test_no_nested_diffs(self):
        d1 = Diff.no_change(1.0)
        d2 = Diff.unknown_change(d1)
        assert not isinstance(d2.get_primal(), Diff)

        assert Diff.static_check_no_change(d1)
        assert not Diff.static_check_no_change(d2)

    def test_tree_diff(self):
        primal_tree = {"a": 1, "b": [2, 3]}
        tangent_tree = {"a": NoChange, "b": [UnknownChange, NoChange]}
        result = Diff.tree_diff(primal_tree, tangent_tree)
        assert isinstance(result["a"], Diff)
        assert isinstance(result["b"][0], Diff)
        assert isinstance(result["b"][1], Diff)
        assert result["a"].get_tangent() == NoChange
        assert result["b"][0].get_tangent() == UnknownChange
        assert result["b"][1].get_tangent() == NoChange

    def test_tree_primal(self):
        tree = {"a": Diff(1, NoChange), "b": [Diff(2, UnknownChange), 3]}
        result = Diff.tree_primal(tree)
        assert result == {"a": 1, "b": [2, 3]}

    def test_tree_tangent(self):
        tree = {"a": 1, "b": [Diff(2, UnknownChange), 3]}
        result = Diff.tree_tangent(tree)

        # note that non-Diffs are marked as UnknownChange, the default tangent value.
        assert result == {"a": NoChange, "b": [UnknownChange, NoChange]}

    def test_static_check_tree_diff(self):
        tree1 = {"a": Diff(1, NoChange), "b": [Diff(2, UnknownChange)]}
        tree2 = {"a": Diff(1, NoChange), "b": [2]}
        assert Diff.static_check_tree_diff(tree1)
        assert not Diff.static_check_tree_diff(tree2)

    def test_static_check_no_change(self):
        tree1 = {"a": Diff(1, NoChange), "b": [Diff(2, NoChange)]}
        tree2 = {"a": Diff(1, NoChange), "b": [Diff(2, UnknownChange)]}
        assert Diff.static_check_no_change(tree1)
        assert not Diff.static_check_no_change(tree2)
