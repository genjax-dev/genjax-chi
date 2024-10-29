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


import functools

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import checkify

from genjax._src.checkify import optional_check
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import FlagOp, tree_choose
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

    # TODO check that these are broadcast-compatible when they come in.
    @staticmethod
    def build(v: "R | Mask[R]", f: Flag | Diff[Flag] = True) -> "Mask[R]":
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
        return Mask.build(v, f).flatten()

    #############
    # Accessors #
    #############

    def flatten(self) -> "R | Mask[R] | None":
        """
        Flatten a Mask instance into its underlying value or None.

        "Flattening" occurs when the flag value is a concrete Boolean (True/False). In these cases, the Mask is simplified to either its raw value or None. If the flag is not concrete (i.e., a symbolic/traced value), the Mask remains intact.

        This method evaluates the mask's flag and returns:
        - None if the flag is concretely False or the value is None
        - The raw value if the flag is concretely True
        - The Mask instance itself if the flag is not concrete

        Returns:
            The flattened result based on the mask's flag state.
        """
        flag = self.primal_flag()
        if FlagOp.concrete_false(flag) or self.value is None:
            return None
        elif FlagOp.concrete_true(flag):
            return self.value
        else:
            return self

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

    def _validate_mask_shapes(self, other: "Mask[R]"):
        # Check that values have same shape
        # Check tree structure matches
        if jtu.tree_structure(self.value) != jtu.tree_structure(other.value):
            raise ValueError("Cannot combine masks with different tree structures!")

        # Check array shapes match exactly (no broadcasting)
        def check_leaf_shapes(x, y):
            x_shape = jnp.shape(x)
            y_shape = jnp.shape(y)
            if x_shape != y_shape:
                raise ValueError(
                    f"Cannot combine masks with different array shapes: {x_shape} vs {y_shape}"
                )
            return None

        jtu.tree_map(check_leaf_shapes, self.value, other.value)

    def _or_idx(self, first: Flag, second: Flag):
        """Converts a pair of flags into an index for selecting between two values.

        This function implements a truth table for selecting between two values based on their flags:

        first | second | output | meaning
        ------+--------+--------+------------------
            0   |   0    |   -1   | neither valid
            1   |   0    |    0   | first valid only
            0   |   1    |    1   | second valid only
            1   |   1    |    0   | both valid for OR, invalid for XOR

        The output index is used to select between the corresponding values:
           0 -> select first value
           1 -> select second value

        Args:
            first: The flag for the first value
            second: The flag for the second value

        Returns:
            An index (-1, 0, or 1) indicating which value to select
        """
        return first + 2 * FlagOp.and_(FlagOp.not_(first), second) - 1

    def __or__(self, other: "Mask[R]") -> "Mask[R]":
        self._validate_mask_shapes(other)

        match self.primal_flag(), other.primal_flag():
            case True, _:
                return self
            case False, _:
                return other
            case self_flag, other_flag:
                idx = self._or_idx(self_flag, other_flag)
                return tree_choose(idx, [self, other])

    def __xor__(self, other: "Mask[R]") -> "Mask[R]":
        self._validate_mask_shapes(other)

        match self.primal_flag(), other.primal_flag():
            case (False, False) | (True, True):
                return Mask.build(self, False)
            case True, False:
                return self
            case False, True:
                return other
            case self_flag, other_flag:
                idx = self._or_idx(self_flag, other_flag)

                # note that `idx` above will choose the correct side for the FF, FT and TF cases,
                # but will equal 0 for TT flags. We use `FlagOp.xor_` to override this flag to equal
                # False, since neither side in the TT case will provide a `False` flag for us.
                chosen = tree_choose(idx, [self.value, other.value])
                return Mask.build(chosen, FlagOp.xor_(self_flag, other_flag))

    @staticmethod
    def or_n(mask: "Mask[R]", *masks: "Mask[R]") -> "Mask[R]":
        """Performs an n-ary OR operation on a sequence of Mask objects.

        Args:
            mask: The first mask to combine
            *masks: Variable number of additional masks to combine with OR

        Returns:
            A new Mask combining all inputs with OR operations
        """
        return functools.reduce(lambda a, b: a | b, masks, mask)

    @staticmethod
    def xor_n(mask: "Mask[R]", *masks: "Mask[R]") -> "Mask[R]":
        """Performs an n-ary XOR operation on a sequence of Mask objects.

        Args:
            mask: The first mask to combine
            *masks: Variable number of additional masks to combine with XOR

        Returns:
            A new Mask combining all inputs with XOR operations
        """
        return functools.reduce(lambda a, b: a ^ b, masks, mask)
