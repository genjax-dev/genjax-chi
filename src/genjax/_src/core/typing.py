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
import jaxtyping as jtyping
import numpy as np
from beartype import BeartypeConf
from beartype import beartype
from plum import dispatch
from plum import parametric


Dataclass = typing.Any
PrettyPrintable = typing.Any
PRNGKey = jtyping.UInt[jtyping.Array, "..."]
FloatArray = jtyping.Float[jtyping.Array, "..."]
BoolArray = jtyping.Bool[jtyping.Array, "..."]
IntArray = jtyping.Int[jtyping.Array, "..."]
Any = typing.Any
Union = typing.Union
Callable = btyping.Callable
Sequence = typing.Sequence
Tuple = btyping.Tuple
Dict = btyping.Dict
List = btyping.List
Iterable = btyping.Iterable
Generator = btyping.Generator
Hashable = btyping.Hashable
FrozenSet = btyping.FrozenSet
Optional = btyping.Optional
Type = btyping.Type
Int = int
Float = float
Bool = bool
String = str
Address = Union[String, Int, Tuple["Address"]]
Value = Any

############
# Generics #
############

Generic = btyping.Generic
TypeVar = btyping.TypeVar

########################################
# Static typechecking from annotations #
########################################

conf = BeartypeConf(is_color=False)
typecheck = beartype(conf=conf)


#################
# Static checks #
#################


def static_check_is_array(v):
    return (
        isinstance(v, jnp.ndarray)
        or isinstance(v, np.ndarray)
        or isinstance(v, jax.core.Tracer)
    )


def static_check_is_concrete(x):
    return not isinstance(x, jax.core.Tracer)


# TODO: the dtype comparison needs to be replaced with something
# more robust.
def static_check_supports_grad(v):
    return static_check_is_array(v) and v.dtype == np.float32


__all__ = [
    "PrettyPrintable",
    "Dataclass",
    "PRNGKey",
    "FloatArray",
    "BoolArray",
    "IntArray",
    "Value",
    "Tuple",
    "Any",
    "Union",
    "Callable",
    "Sequence",
    "Dict",
    "List",
    "Int",
    "Bool",
    "Float",
    "Generator",
    "Iterable",
    "Type",
    "Generic",
    "TypeVar",
    "static_check_is_concrete",
    "static_check_is_array",
    "static_check_supports_grad",
    "typecheck",
    "dispatch",
    "parametric",
]
