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
This module implements a generative function combinator which allows a recursive grammar-like pattern for generative functions.
"""

from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Tuple

import jax
import jax.numpy as jnp

from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Trace
from genjax.core.masks import BooleanMask

#####
# RecurseTrace
#####


@dataclass
class RecurseTrace(Trace):
    pass


#####
# RecurseCombinator
#####


@dataclass
class RecurseCombinator(GenerativeFunction):
    pass
