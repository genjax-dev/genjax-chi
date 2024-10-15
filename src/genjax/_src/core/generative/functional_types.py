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

from typing import overload

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import checkify

from genjax._src.checkify import optional_check
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import (
    FlagOp,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Array,
    ArrayLike,
    Flag,
    Generic,
    TypeVar,
)

R = TypeVar("R")

#########################
# Masking and sum types #
#########################


@Pytree.dataclass(match_args=True)
class Mask(Generic[R], Pytree):
    """The `Mask` datatype wraps a value in a Boolean flag which denotes whether the data is valid or invalid to use in inference computations.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `ChoiceMap` leaves, and participate in generative and inference computations (like scores, and importance weights or density ratios). Invalid data **should** be considered unusable, and should be handled with care.

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    ## Encountering `Mask` in your computation

    When users see `Mask` in their computations, they are expected to interact with them by either:

    * Unmasking them using the `Mask.unmask` interface, a potentially unsafe operation.

    * Destructuring them manually, and handling the cases.

    ## Usage of invalid data

    If you use invalid `Mask(data, False)` data in inference computations, you may encounter silently incorrect results.
    """

    value: R
    flag: Flag | Diff[Flag]

    @overload
    @staticmethod
    def maybe(v: "Mask[R]", f: Flag) -> "Mask[R]": ...

    @overload
    @staticmethod
    def maybe(v: R, f: Flag | Diff[Flag]) -> "Mask[R]": ...

    @staticmethod
    def maybe(v: "R | Mask[R]", f: Flag | Diff[Flag]) -> "Mask[R]":
        match v:
            case Mask(value, g):
                assert not isinstance(f, Diff) and not isinstance(g, Diff)
                return Mask[R](value, FlagOp.and_(f, g))
            case _:
                return Mask[R](v, f)

    @staticmethod
    def maybe_none(v: "R | Mask[R]", f: Flag) -> "R | Mask[R] | None":
        if v is None or FlagOp.concrete_false(f):
            return None
        elif FlagOp.concrete_true(f):
            return v
        else:
            return Mask.maybe(v, f)

    ######################
    # Masking interfaces #
    ######################

    def unsafe_unmask(self) -> R:
        """
        Unsafe version of unmask -- should only be used internally, or carefully.
        """
        return self.value

    def unmask(self, default: R | None = None) -> R:
        """
        Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` wraps valid data at runtime.

        Args:
            default: An optional default value to return if the mask is invalid.

        Returns:
            The unmasked value if valid, or the default value if provided and the mask is invalid.
        """
        if default is None:

            def _check():
                checkify.check(
                    self.primal_flag(),
                    "Attempted to unmask when a mask flag is False: the masked value is invalid.\n",
                )

            optional_check(_check)
            return self.unsafe_unmask()
        else:

            def inner(true_v: ArrayLike, false_v: ArrayLike) -> Array:
                return jnp.where(self.primal_flag(), true_v, false_v)

            return jtu.tree_map(inner, self.value, default)

    def primal_flag(self) -> Flag:
        match self.flag:
            case Diff(primal, _):
                return primal
            case _:
                return self.flag
