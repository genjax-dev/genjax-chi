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
"""This module provides the core functionality and JAX compatibility layer
which several `GenJAX` generative function and inference modules are built on
top of.

It contains (truncated, and in no particular order):

* Core data types for the associated data types of generative functions.

* Utility abstract data types (mixins) for automatically registering class definitions as valid `Pytree` method implementors (guaranteeing `flatten`/`unflatten` compatibility across JAX transform boundaries). For more information, see [Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html).

* Transformation interpreters: interpreter-based transformations which operate on `ClosedJaxpr` instances, as well as staging functionality for staging out computations to `ClosedJaxpr` instances. The core interpreters are written in a mixed initial / final style. The application of all interpreters are JAX compatible, meaning that the application of any interpreter can be staged out to eliminate the interpreter overhead.
"""
