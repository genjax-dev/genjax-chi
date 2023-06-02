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

"""
The `combinators` module exposes _generative function combinators_, generative functions which accept other generative functions as configuration arguments, and implement structured patterns of control flow (as well as other types of modifications) as their generative function interface implementations.

GenJAX features several standard combinators:

* `UnfoldCombinator` - which exposes a scan-like pattern for generative computation in a state space pattern via implementations utilizing `jax.lax.scan`.
* `MapCombinator` - which exposes generative vectorization over input arguments, whose implementation utilizes `jax.vmap`.
* `SwitchCombinator` - which exposes stochastic branching patterns utilizing `jax.lax.switch`.
"""
