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
throughout the codebase.

Type annotations in the codebase are exported out of this module for
consistency.
"""

import typing

import beartype.typing as btyping
import jax
import jax.numpy as jnp
import numpy as np
from beartype import BeartypeConf
from beartype import beartype
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import UInt


conf = BeartypeConf(is_color=False)
typecheck = beartype(conf=conf)

PRNGKey = UInt[Array, "..."]
PrettyPrintable = typing.Any
Dataclass = typing.Any
FloatArray = typing.Union[float, Float[Array, "..."]]
BoolArray = typing.Union[bool, Bool[Array, "..."]]
IntArray = typing.Union[int, Int[Array, "..."]]
Tuple = btyping.Tuple
Any = typing.Any
Union = typing.Union
Callable = typing.Callable
Sequence = typing.Sequence
Dict = btyping.Dict
List = btyping.List
Int = int


def static_check_is_array(v):
    return (
        isinstance(v, jnp.ndarray)
        or isinstance(v, np.ndarray)
        or isinstance(v, jax.core.Tracer)
    )


# TODO: the dtype comparison needs to be replaced with something
# more robust.
def static_check_grad(v):
    return static_check_is_array(v) and v.dtype == np.float32


__all__ = [
    "PRNGKey",
    "PrettyPrintable",
    "Dataclass",
    "FloatArray",
    "BoolArray",
    "IntArray",
    "Int",
    "Tuple",
    "Any",
    "Union",
    "Callable",
    "Sequence",
    "Dict",
    "List",
    "static_check_is_array",
]
