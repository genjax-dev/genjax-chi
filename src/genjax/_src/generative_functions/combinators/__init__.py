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
"""The `combinators` module exposes _generative function combinators_: generative
functions which accept other generative functions as configuration arguments, and
implement their own generative function interfaces using structured patterns of control
flow (and other types of useful modifications). If one thinks of a control flow
primitive as an operation on deterministic types, a combinator can be thought of as
lifting the operation to support generative function semantics.

GenJAX exposes several combinators:

* [`MaskedCombinator`](masked.md) - which can mask a generative computation based on a runtime determined `BoolArray` argument.
* [`MapCombinator`](map.md) - which exposes generative vectorization over input arguments. The implementation essentially wraps [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) into the interfaces.
* [`UnfoldCombinator`](unfold.md) - which exposes a scan-like pattern for generative computation in a state space pattern, by utilizing the control flow primitive [`jax.lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html).
* [`SwitchCombinator`](switch.md) - which exposes stochastic branching patterns, by utilizing the control flow primitive [`jax.lax.switch`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html).
"""
