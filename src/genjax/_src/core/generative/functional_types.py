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
    BoolArray,
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
    """The `Mask` datatype wraps a value in a BoolArray flag which denotes whether the data is valid or invalid to use in inference computations.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `ChoiceMap` leaves, and participate in generative and inference computations (like scores, and importance weights or density ratios). Invalid data **should** be considered unusable, and should be handled with care.

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    Note that the flag needs to be broadcast-compatible with the value, or with ALL the value's leaves if the value is a pytree. For more information on broadcasting semantics, refer to the NumPy documentation on broadcasting: [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

    ## Encountering `Mask` in your computation

    When users see `Mask` in their computations, they are expected to interact with them by either:

    * Unmasking them using the `Mask.unmask` interface, a potentially unsafe operation.

    * Destructuring them manually, and handling the cases.

    ## Usage of invalid data

    If you use invalid `Mask(data, False)` data in inference computations, you may encounter silently incorrect results.
    """

    value: R
    flag: Flag | Diff[Flag]

    ################
    # Constructors #
    ################

    @staticmethod
    def build(v: "R | Mask[R]", f: Flag | Diff[Flag]) -> "Mask[R]":
        """
        Create a Mask instance, potentially from an existing Mask or a raw value.

        This method allows for the creation of a new Mask or the modification of an existing one. If the input is already a Mask, it combines the new flag with the existing one using a logical AND operation.

        Args:
            v: The value to be masked. Can be a raw value or an existing Mask.
            f: The flag to be applied to the value.

        Returns:
            A new Mask instance with the given value and flag.

        Note:
            If `v` is already a Mask, the new flag is combined with the existing one using a logical AND, ensuring that the resulting Mask is only valid if both input flags are valid.
        """
        match v:
            case Mask(value, g):
                assert not isinstance(f, Diff) and not isinstance(g, Diff)
                return Mask[R](value, FlagOp.and_(f, g))
            case _:
                return Mask[R](v, f)

    @staticmethod
    def maybe_mask(v: "R | Mask[R] | None", f: Flag) -> "R | Mask[R] | None":
        """
        Create a Mask instance or return the original value based on the flag.

        This method is similar to `build`, but it handles concrete flag values differently. For concrete True flags, it returns the original value without wrapping it in a Mask. For concrete False flags, it returns None. For non-concrete flags, it creates a new Mask instance.

        Args:
            v: The value to be potentially masked. Can be a raw value or an existing Mask.
            f: The flag to be applied to the value.

        Returns:
            - The original value `v` if `f` is concretely True.
            - None if `f` is concretely False.
            - A new Mask instance with the given value and flag if `f` is not concrete.
        """
        if FlagOp.concrete_false(f):
            return None
        elif FlagOp.concrete_true(f):
            return v
        else:
            return Mask.build(v, f)

    #############
    # Accessors #
    #############

    def unmask(self, default: R | None = None) -> R:
        """
        Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics if no default value is provided. It is only valid if the `Mask` wraps valid data at runtime, or if a default value is supplied.

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
            return self.value
        else:

            def inner(true_v: ArrayLike, false_v: ArrayLike) -> Array:
                return jnp.where(self.primal_flag(), true_v, false_v)

            import jax

            jax.lax.broadcast_shapes
            return jtu.tree_map(inner, self.value, default)

    def primal_flag(self) -> Flag:
        """
        Returns the primal flag of the mask.

        This method retrieves the primal (non-`Diff`-wrapped) flag value. If the flag
        is a Diff type (which contains both primal and tangent components), it returns
        the primal component. Otherwise, it returns the flag as is.

        Returns:
            The primal flag value.
        """
        match self.flag:
            case Diff(primal, _):
                return primal
            case flag:
                return flag

        ###############
        # Combinators #
        ###############

    @staticmethod
    def binary_or(a: "Mask[R]", b: "Mask[R]") -> "Mask[R]":
        check1 = a.primal_flag()
        check2 = b.primal_flag()

        # Short-circuit for concrete booleans
        if FlagOp.concrete_true(check1):
            return a
        if FlagOp.concrete_true(check2):
            return b
        if FlagOp.concrete_false(check1) and FlagOp.concrete_false(check2):
            return Mask(
                b.value, False
            )  # Both false, return either value with False flag

        # Combine the flags
        or_flag = FlagOp.or_(check1, check2)

        # Choose the value based on the flags
        def choose_value(a_val, b_val):
            return FlagOp.where(or_flag, a_val, b_val)

        chosen_value = jtu.tree_map(choose_value, a.value, b.value)

        return Mask(chosen_value, or_flag)

    @staticmethod
    def _or_many(*masks: "Mask[R]") -> "Mask[R]":
        flags = [mask.primal_flag() for mask in masks]

        # Check if all flags are concrete booleans
        if all(isinstance(flag, bool) for flag in flags):
            # Short-circuit for concrete booleans
            for mask, flag in zip(masks, flags):
                if flag:
                    return mask
            # If all flags are False, return the last mask with a False flag
            return Mask(masks[-1].value, False)

        # If not all concrete, proceed with the general case
        values_and_flags = jnp.array([
            (mask.value, mask.primal_flag()) for mask in masks
        ])
        values_array, flags_array = values_and_flags[:, 0], values_and_flags[:, 1]
        combined_flag = jnp.any(flags_array)

        def choose_value(values: Array, flags: BoolArray):
            first_true_index = jnp.argmax(flags)
            # TODO this is broken under jit.
            return jtu.tree_map(lambda *vs: vs[first_true_index], *values)

        chosen_value = choose_value(values_array, flags_array)

        return Mask(chosen_value, combined_flag)
