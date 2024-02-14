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
"""This module contains several standard generative function classes useful for
structuring probabilistic programs.

* The [`distributions`](./distributions/) module provides generative function wrappers for standard distributions from TensorFlow Probability Distributions (`tfd`), as well as custom distributions. All of the distributions exported from this module are `GenerativeFunction`.
* The [`static`](./static.md) module contains a programmatic generative function language which utilizes restricted Python programs (meaning, JAX traceable and transformable) as the source language for defining generative functions.
* The [`combinators`](./combinators/) module contains combinators which support transforming generative functions into new ones with structured control flow patterns of computation, and other effects.
* The [`interpreted`](./interpreted.md) module exposes an expressive (allowed to use arbitrary Python) generative function language for sketching models and for learning GenJAX. **Note: this language cannot be used compositionally (as a callee) with the other languages described above**.
"""
