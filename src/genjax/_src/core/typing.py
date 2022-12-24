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

"""This module contains a set of types and type aliases which are used
throughout the codebase."""

from typing import Any
from typing import Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int


PRNGKey = Int[Array, "..."]
PrettyPrintable = Any
Dataclass = Any
FloatArray = Union[float, Float[Array, "..."]]
BoolArray = Union[bool, Bool[Array, "..."]]
IntArray = Union[int, Int[Array, "..."]]


def static_check_is_array(v):
    isinstance(v, jnp.ndarray) or isinstance(v, np.ndarray)
