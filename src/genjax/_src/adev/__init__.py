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
"""This module provides an implementation of [ADEV: Sound Automatic Differentiation of Expected Values](https://dl.acm.org/doi/abs/10.1145/3571198), an AD algorithm which derivates forward mode derivative estimators for programs which denote expectations.

Our implementation tightly integrates with JAX's AD machinery, and extends the system described in the original paper with a reverse mode implementation, by utilizing JAX's support for automatically deriving a reverse mode from a forward mode (via [You Only Linearize Once: Tangents Transpose to Gradients](https://dl.acm.org/doi/abs/10.1145/3571236)).

Our implementation of the D{.} transformation from ADEV is structured around a separation of concerns of ADEV's transformation into two interpreters:

1. There's a partial CPS interpreter, which transforms a `Jaxpr` into a CPS-transformed `Jaxpr`, where certain primitives get access to their continuations.

2. There's a forward mode interpreter, which implements ADEV's forward mode AD for CPS-transformed `Jaxpr`s.
"""
