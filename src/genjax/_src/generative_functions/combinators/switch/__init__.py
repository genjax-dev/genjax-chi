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
"""The `SwitchCombinator` is a generative function combinator which allows
branching control flow, with branches that are generative functions.

The freedoms: each branch generative function can produce choice maps with heterogeneous address/value spaces, which don't have to be Pytree type equal to the choice maps of other branches.

There are restrictions: generative functions which are passed in as branch generative functions to `SwitchCombinator`
must accept the same argument types (Pytree type equality), and return the same return type (Pytree type equality).

The internal choice maps for the branch generative functions
can have different shape/dtype choices.

Internally, `SwitchCombinator` implements a scheme to efficiently share `(shape, dtype)` storage across branches.
"""
