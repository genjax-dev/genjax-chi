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
This module provides a builtin modeling language built on top of JAX interpreters.

It exposes a set of JAX primitives which allow compositional 
construction of generative programs.
These programs can utilize other generative functions 
(for example, see the `distributions`) library.
"""

from .intrinsics import trace
from .handlers import *
from .datatypes import *
from .tracetypes import *
from .tree import *
from .shape_analysis import *

__all__ = [
    "trace",
    "Trace",
    "ChoiceMap",
    "Selection",
    "BuiltinGenerativeFunction",
    "BuiltinTrace",
    "BuiltinChoiceMap",
    "BuiltinSelection",
    "BuiltinTraceType",
    "trace_shape",
    "trace_shape_no_toplevel",
    "choice_map_shape",
]
