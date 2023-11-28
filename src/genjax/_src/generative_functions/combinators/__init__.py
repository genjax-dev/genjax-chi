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
"""The `combinators` module exposes _generative function combinators_:
generative functions which accept other generative functions as configuration
arguments, and implement their own interfaces using structured patterns of
control flow (as well as other types of modifications). If one thinks of a
control flow primitive as an operation on deterministic types, a combinator can
be thought of as lifting the operation to support generative function
semantics.

GenJAX exposes several standard combinators:

* `MaskedCombinator` - which can "mask" a generative computation based on a runtime determined `BoolArray` argument.
* `MapCombinator` - which exposes generative vectorization over input arguments. The implementation utilizes `jax.vmap`.
* `RepeatCombinator` - which wraps `MapCombinator` to support vectorized IID sampling for fixed input arguments.
* `UnfoldCombinator` - which exposes a scan-like pattern for generative computation in a state space pattern, by utilizing `jax.lax.scan`.
* `SwitchCombinator` - which exposes stochastic branching patterns utilizing `jax.lax.switch`.
"""
