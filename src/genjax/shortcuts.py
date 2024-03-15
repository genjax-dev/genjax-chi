# Copyright 2024 MIT Probabilistic Computing Project
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

from deprecated import deprecated

# Future deprecated APIs.
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.shortcuts import (
    choice,
    choice_map,
    indexed_choice_map,
    select,
    vector_choice_map,
)


@deprecated(
    reason="The tree_diff prefixed functions are now accessible via `Diff` static methods directly e.g. `Diff.tree_diff`"
)
def tree_diff(v, t):
    return Diff.tree_diff(v, t)


@deprecated(
    reason="The tree_diff prefixed functions are now accessible via `Diff` static methods directly e.g. `Diff.tree_diff_no_change`"
)
def tree_diff_no_change(v):
    return Diff.tree_diff_no_change(v)


@deprecated(
    reason="The tree_diff prefixed functions are now accessible via `Diff` static methods directly e.g. `Diff.tree_diff_unknown_change`"
)
def tree_diff_unknown_change(v):
    return Diff.tree_diff_unknown_change(v)


@deprecated(
    reason="The tree_diff prefixed functions are now accessible via `Diff` static methods directly e.g. `Diff.tree_primal`"
)
def tree_diff_primal(v):
    return Diff.tree_primal(v)


@deprecated(
    reason="The tree_diff prefixed functions are now accessible via `Diff` static methods directly e.g. `Diff.tree_tangent`"
)
def tree_diff_tangent(v):
    return Diff.tree_tangent(v)


__all__ = [
    "choice",
    "choice_map",
    "indexed_choice_map",
    "select",
    "vector_choice_map",
    "tree_diff",
    "tree_diff_no_change",
    "tree_diff_unknown_change",
    "tree_diff_primal",
    "tree_diff_tangent",
]
