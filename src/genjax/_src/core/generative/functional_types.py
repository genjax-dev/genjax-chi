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

import jax.numpy as jnp
from jax.experimental import checkify
from jax.tree_util import tree_map

from genjax._src.checkify import optional_check
from genjax._src.core.interpreters.staging import (
    Flag,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    ArrayLike,
    Generic,
    Int,
    TypeVar,
)

R = TypeVar("R")

#########################
# Masking and sum types #
#########################


@Pytree.dataclass(match_args=True)
class Mask(Generic[R], Pytree):
    """The `Mask` datatype wraps a value in a Boolean flag which denotes whether the data is valid or invalid to use in inference computations.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `Sample` instances, and participate in generative and inference computations (like scores, and importance weights or density ratios). Invalid data **should** be considered unusable, and should be handled with care.

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    ## Encountering `Mask` in your computation

    When users see `Mask` in their computations, they are expected to interact with them by either:

    * Unmasking them using the `Mask.unmask` interface, a potentially unsafe operation.

    * Destructuring them manually, and handling the cases.

    ## Usage of invalid data

    If you use invalid `Mask(False, data)` data in inference computations, you may encounter silently incorrect results.
    """

    flag: Flag
    value: R

    @classmethod
    def maybe(cls, f: Flag, v: "R | Mask[R]") -> "Mask[R]":
        match v:
            case Mask(flag, value):
                return Mask[R](f.and_(flag), value)
            case _:
                return Mask[R](f, v)

    @classmethod
    def maybe_none(cls, f: Flag, v: "R | Mask[R]") -> "R | Mask[R] | None":
        if v is None or f.concrete_false():
            return None
        elif f.concrete_true():
            return v
        else:
            return Mask.maybe(f, v)

    ######################
    # Masking interfaces #
    ######################

    def unsafe_unmask(self) -> R:
        """
        Unsafe version of unmask -- should only be used internally, or carefully.
        """
        return self.value

    def unmask(self) -> R:
        """
        Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` wraps valid data at runtime.
        """

        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            checkify.check(
                self.flag.f,
                "Attempted to unmask when a mask flag is False: the masked value is invalid.\n",
            )

        optional_check(_check)
        return self.unsafe_unmask()


def staged_choose(
    idx: ArrayLike,
    pytrees: list[R],
) -> R:
    """
    Version of `jax.numpy.choose` that

    - acts on lists of both `ArrayLike` and `Pytree` instances
    - acts like `vs[idx]` if `idx` is of type `int`.

    In the case of heterogenous types in `vs`, `staged_choose` will attempt to cast, or error if casting isn't possible. (mixed `bool` and `int` entries in `vs` will result in the cast of selected `bool` to `int`, for example.).

    Args:
        idx: The index used to select a value from `vs`.
        vs: A list of `Pytree` or `ArrayLike` values to choose from.

    Returns:
        The selected value from the list.
    """

    def inner(*vs: ArrayLike) -> ArrayLike:
        # Computing `result` above the branch allows us to:
        # - catch incompatible types / shapes in the result
        # - in the case of compatible types requiring casts (like bool => int),
        #   result's dtype tells us the final type.
        result = jnp.choose(idx, vs, mode="wrap")
        if isinstance(idx, Int):
            return jnp.asarray(vs[idx % len(vs)], dtype=result.dtype)
        else:
            return result

    return tree_map(inner, *pytrees)
