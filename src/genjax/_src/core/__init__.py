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

"""This module provides the core functionality which the `GenJAX` modeling and
inference modules are buillt on top of, as well as utility functionality for
coercing class definitions to valid `Pytree` method implementors (guaranteeing
compatibility with JAX's serializing/deserializing to `Pytree` instances).

This module also exports some interpreter-based transformations on `Jaxpr`
instances. Interpreters are all written in initial style - they operate on `ClosedJaxpr` instances - but they are JAX compatible, meaning they can be staged out for zero runtime cost.
"""

from .callback import *
from .datatypes import *
from .diff_rules import *
from .hashabledict import *
from .interpreters import *
from .masks import *
from .pytree import *
from .specialization import *
from .staging import *
from .sumtree import *
from .tracetypes import *
from .tree import *
