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
"""This module contains a set of types and type aliases which are used throughout the
codebase.

Type annotations in the codebase are exported out of this module for consistency.
"""

from typing import Annotated  # noqa: I001
from types import EllipsisType

import beartype.typing as btyping
from jax import core as jc
import jax.numpy as jnp
import jaxtyping as jtyping
import numpy as np
from beartype.vale import Is
import sys

if sys.version_info >= (3, 11, 0):
    from typing import Self
else:
    from typing_extensions import Self

Any = btyping.Any
PRNGKey = jtyping.PRNGKeyArray
Array = jtyping.Array
ArrayLike = jtyping.ArrayLike
IntArray = jtyping.Int[jtyping.Array, "..."]
FloatArray = jtyping.Float[jtyping.Array, "..."]
BoolArray = jtyping.Bool[jtyping.Array, "..."]
Callable = btyping.Callable
Sequence = btyping.Sequence

# JAX Type alias.
InAxes = int | None | Sequence[Any]

# Types of Python literals.
Int = int
Float = float
Bool = bool
String = str

Value = Any

#################################
# Trace-time-checked primitives #
#################################

ScalarShaped = Is[lambda arr: jnp.array(arr, copy=False).shape == ()]
ScalarBool = Annotated[Bool | BoolArray, ScalarShaped]

############
# Generics #
############

Generic = btyping.Generic
TypeVar = btyping.TypeVar
ParamSpec = btyping.ParamSpec


#################
# Static checks #
#################


def static_check_is_array(v: Any) -> Bool:
    return (
        isinstance(v, jnp.ndarray)
        or isinstance(v, np.ndarray)
        or isinstance(v, jc.Tracer)
    )


def static_check_is_concrete(x: Any) -> bool:
    return not isinstance(x, jc.Tracer)


# TODO: the dtype comparison needs to be replaced with something
# more robust.
def static_check_supports_grad(v):
    return static_check_is_array(v) and v.dtype == np.float32


def static_check_shape_dtype_equivalence(vs: list[Array]) -> Bool:
    shape_dtypes = [(v.shape, v.dtype) for v in vs]
    num_unique = set(shape_dtypes)
    return len(num_unique) == 1


__all__ = [
    "Annotated",
    "Any",
    "Array",
    "ArrayLike",
    "Bool",
    "BoolArray",
    "Callable",
    "EllipsisType",
    "Float",
    "FloatArray",
    "Generic",
    "InAxes",
    "Int",
    "IntArray",
    "Is",
    "PRNGKey",
    "ParamSpec",
    "ScalarBool",
    "ScalarShaped",
    "Self",
    "Sequence",
    "TypeVar",
    "Value",
    "static_check_is_array",
    "static_check_is_concrete",
    "static_check_shape_dtype_equivalence",
    "static_check_supports_grad",
]
