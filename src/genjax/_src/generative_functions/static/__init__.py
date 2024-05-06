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
"""This module provides a program-like generative function language. The generative
function interfaces are implemented for the objects in this language using
transformations applied via `Jaxpr` interpreters.

To support sequencing of probabilistic computations as a capability in the modeling language, this language also exposes custom JAX primitives which denote invocation of other generative functions (as callees): model programs written in this language can
utilize other generative functions using the exposed `trace` (or the syntactic sugared version) to create hierarchical patterns of generative computation.
"""

from .static_gen_fn import StaticGenerativeFunction, static_gen_fn
from .static_transforms import AddressReuse, StaticAddressJAX, trace, trace_p

__all__ = [
    "StaticAddressJAX",
    "AddressReuse",
    "StaticGenerativeFunction",
    "static_gen_fn",
    "trace_p",
    "trace",
]
