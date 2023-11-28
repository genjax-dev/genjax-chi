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
"""This module contains several standard generative function classes useful for
structuring probabilistic programs.

* The `distributions` module exports standard distributions from TensorFlow Probability Distributions (`tfd`), as well as custom distributions.
* The `static` module contains a JAX compatible (meaning, traceable and transformable) language for defining generative functions from Python functions.
* The `combinators` module contains combinators which support transforming generative functions into new ones with structured control flow patterns of computation, and other effects.
* The `interpreted` module exposes an expressive (allowed to use arbitrary Python) language for sketching models and pedagogy. **This language cannot be used compositionally with the languaged described above**.
"""
