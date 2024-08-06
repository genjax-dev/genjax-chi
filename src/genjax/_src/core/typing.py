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
"""This module contains a set of types and type aliases which are used
throughout the codebase.

Type annotations in the codebase are exported out of this module for
consistency.

"""

from typing import Annotated  # noqa: I001
from types import EllipsisType

import beartype.typing as btyping
from jax import core as jc
import jax.numpy as jnp
import jaxtyping as jtyping
import numpy as np
from beartype import BeartypeConf, beartype
from beartype.vale import Is

Any = btyping.Any
PRNGKey = jtyping.PRNGKeyArray
Array = jtyping.Array
ArrayLike = jtyping.ArrayLike
IntArray = jtyping.Int[jtyping.Array, "..."]
FloatArray = jtyping.Float[jtyping.Array, "..."]
BoolArray = jtyping.Bool[jtyping.Array, "..."]
Callable = btyping.Callable
Sequence = btyping.Sequence
Optional = btyping.Optional
Type = btyping.Type
Literal = btyping.Literal
Protocol = btyping.Protocol
Self = btyping.Self

runtime_checkable = btyping.runtime_checkable
overload = btyping.overload

# JAX Type alias.
InAxes = int | None | Sequence[Any]


String = str

Value = Any

#################################
# Trace-time-checked primitives #
#################################

ScalarShaped = Is[lambda arr: jnp.array(arr, copy=False).shape == ()]
ScalarBool = Annotated[bool | BoolArray, ScalarShaped]

############
# Generics #
############

Generic = btyping.Generic
TypeVar = btyping.TypeVar
ParamSpec = btyping.ParamSpec

########################################
# Static typechecking from annotations #
########################################

global_conf = BeartypeConf(
    is_color=True,
    is_debug=False,
    is_pep484_tower=True,
    violation_type=TypeError,
)


def typecheck_with_config(**kwargs):
    return beartype(conf=BeartypeConf(**kwargs))


typecheck = beartype(conf=global_conf)
"""Accepts a function, and returns a function which uses `beartype` to perform
type checking.

Examples:
    The below examples are roughly what you should expect to see if you trip over `beartype` via `genjax.typing.typecheck`.

    ```python exec="yes" source="material-block" session="core"
    from genjax.typing import typecheck_with_config as typecheck

    def f(x: int) -> int:
        return x + 1.0

    try:
        f(1.0)
    except TypeError as e:
        print("TypeError: ", e)
    ```

    This also works for the return values:

    ```python exec="yes" source="material-block" session="core"
    def f(x: int) -> int:
        return x + 1.0

    try:
        f(1)
    except TypeError as e:
        print("TypeError: ", e)
    ```

"""

#################
# Static checks #
#################


def static_check_is_array(v: Any) -> bool:
    return (
        isinstance(v, jnp.ndarray)
        or isinstance(v, np.ndarray)
        or isinstance(v, jc.Tracer)
    )


def static_check_is_concrete(x: Any) -> bool:
    return not isinstance(x, jc.Tracer)


def static_check_bool(x: Any) -> bool:
    return static_check_is_concrete(x) and isinstance(x, bool)


# TODO: the dtype comparison needs to be replaced with something
# more robust.
def static_check_supports_grad(v):
    return static_check_is_array(v) and v.dtype == np.float32


def static_check_shape_dtype_equivalence(vs: list[Array]) -> bool:
    shape_dtypes = [(v.shape, v.dtype) for v in vs]
    num_unique = set(shape_dtypes)
    return len(num_unique) == 1


__all__ = [
    "Annotated",
    "Any",
    "Array",
    "ArrayLike",
    "BoolArray",
    "Callable",
    "EllipsisType",
    "Float",
    "FloatArray",
    "Generic",
    "InAxes",
    "IntArray",
    "Is",
    "Literal",
    "PRNGKey",
    "ParamSpec",
    "Protocol",
    "ScalarBool",
    "ScalarShaped",
    "Self",
    "Sequence",
    "Type",
    "TypeVar",
    "Value",
    "overload",
    "runtime_checkable",
    "static_check_is_array",
    "static_check_is_concrete",
    "static_check_shape_dtype_equivalence",
    "static_check_supports_grad",
    "typecheck",
    "typecheck_with_config",
]
