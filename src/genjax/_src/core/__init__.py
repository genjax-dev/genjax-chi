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

"""This module provides the core functionality and JAX compatibility layer
which the `GenJAX` generative function modeling and inference modules are built
on top of. It contains (truncated, and in no particular order):

* Core [Gen](https://www.gen.dev/) associated data types for generative functions.

* Utility functionality for automatically registering class definitions as valid `Pytree` method implementors (guaranteeing `flatten`/`unflatten` compatibility across JAX transform boundaries). For more information, see [Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html).

* Staging functionality that allows linear lifting of pure, numerical Python programs to `ClosedJaxpr` instances.

* Transformation interpreters: interpreter-based transformations on which operate on `ClosedJaxpr` instances. Interpreters are all written in initial style - they operate on `ClosedJaxpr` instances, and don't implement their own custom `jax.Tracer` types - but they are JAX compatible, implying that they can be staged out for zero runtime cost.

* Masking functionality which allows active/inactive flagging of data - useful when branching patterns of computation require uncertainty in whether or not data is active with respect to a generative computation.
"""
