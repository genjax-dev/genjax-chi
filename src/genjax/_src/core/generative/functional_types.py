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
from abc import abstractmethod

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import checkify

from genjax._src.checkify import optional_check
from genjax._src.core.generative.choice_map import (
    Address,
    AddressComponent,
    DynamicAddressComponent,
    K_addr,
    Selection,
    StaticAddressComponent,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import FlagOp, tree_choose
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Array,
    ArrayLike,
    Callable,
    Final,
    Flag,
    Generic,
    IntArray,
    TypeVar,
)

R = TypeVar("R")


#########################
# Masking and sum types #
#########################


@Pytree.dataclass(match_args=True, init=False)
class Mask(Generic[R], Pytree):
    """The `Mask` datatype wraps a value in a Boolean flag which denotes whether the data is valid or invalid to use in inference computations.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `ChoiceMap` leaves, and participate in generative and inference computations (like scores, and importance weights or density ratios). A Mask with a False flag **should** be considered unusable, and should be handled with care.

    If a `flag` has a non-scalar shape, that implies that the mask is vectorized, and that the `ArrayLike` value, or each leaf in the pytree, must have the flag's shape as its prefix (i.e., must have been created with a `jax.vmap` call or via a GenJAX `vmap` combinator).

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

    def __init__(self, value: R, flag: Flag | Diff[Flag] = True) -> None:
        assert not isinstance(value, Mask), (
            f"Mask should not be instantiated with another Mask! found {value}"
        )
        Mask._validate_init(value, flag)

        self.value, self.flag = value, flag  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def _validate_init(value: R, flag: Flag | Diff[Flag]) -> None:
        """Validates that non-scalar flags are only used with vectorized masks.

        When a flag has a non-scalar shape (e.g. shape (3,)), this indicates the mask is vectorized.
        In this case, each leaf value in the pytree must have the flag's shape as a prefix of its own shape.
        For example, if flag has shape (3,), then array leaves must have shapes like (3,), (3,4), (3,2,1) etc.

        This ensures that vectorized flags properly align with vectorized data.

        Args:
            value: The value to be masked, can be a pytree
            flag: The flag to apply, either a scalar or array flag

        Raises:
            ValueError: If a non-scalar flag's shape is not a prefix of all leaf value shapes
        """
        flag = flag.get_primal() if isinstance(flag, Diff) else flag
        f_shape = jnp.shape(flag)
        if f_shape == ():
            return None

        leaf_shapes = [jnp.shape(leaf) for leaf in jtu.tree_leaves(value)]
        prefix_len = len(f_shape)

        for shape in leaf_shapes:
            if shape[:prefix_len] != f_shape:
                raise ValueError(
                    f"Vectorized flag {flag}'s shape {f_shape} must be a prefix of all leaf shapes. Found {shape}."
                )

    @staticmethod
    def _validate_leaf_shapes(this: R, other: R):
        """Validates that two values have matching shapes at each leaf.

        Used by __or__, __xor__ etc. to ensure we only combine masks with values whose leaves have matching shapes.
        Broadcasting is not supported - array shapes must match exactly.

        Args:
            this: First value to compare
            other: Second value to compare

        Raises:
            ValueError: If any leaf shapes don't match exactly
        """

        # Check array shapes match exactly (no broadcasting)
        def check_leaf_shapes(x, y):
            x_shape = jnp.shape(x)
            y_shape = jnp.shape(y)
            if x_shape != y_shape:
                raise ValueError(
                    f"Cannot combine masks with different array shapes: {x_shape} vs {y_shape}"
                )
            return None

        jtu.tree_map(check_leaf_shapes, this, other)

    def _validate_mask_shapes(self, other: "Mask[R]") -> None:
        """Used by __or__, __xor__ etc. to ensure we only combine masks with matching pytree shape and matching leaf shapes."""
        if jtu.tree_structure(self.value) != jtu.tree_structure(other.value):
            raise ValueError("Cannot combine masks with different tree structures!")

        Mask._validate_leaf_shapes(self, other)
        return None

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
                assert FlagOp.is_scalar(f) or (jnp.shape(f) == jnp.shape(g)), (
                    f"Can't build a Mask with non-matching Flag shapes {jnp.shape(f)} and {jnp.shape(g)}"
                )
                return Mask[R](value, FlagOp.and_(f, g))
            case _:
                return Mask[R](v, f)

    @staticmethod
    def maybe_mask(v: "R | Mask[R]", f: Flag) -> "R | Mask[R] | None":
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

    def __getitem__(self, path) -> "Mask[R]":
        path = path if isinstance(path, tuple) else (path,)

        f = self.primal_flag()
        if isinstance(f, Array) and f.shape:
            # A non-scalar flag must have been produced via vectorization. Because a scalar flag can
            # wrap a non-scalar value, only use the vectorized components of the path to index into the flag...
            f = f[path[: len(f.shape)]]

        # but the use full path to index into the value.
        v_idx = jtu.tree_map(lambda v: v[path], self.value)

        # Reconstruct Diff if needed
        if isinstance(self.flag, Diff):
            f = Diff(f, self.flag.tangent)

        return Mask.build(v_idx, f)

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
        if FlagOp.concrete_false(flag):
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
                    jnp.all(self.primal_flag()),
                    "Attempted to unmask when a mask flag (or some flag in a vectorized mask) is False: the unmasked value is invalid.\n",
                )

            optional_check(_check)
            return self.value
        else:

            def inner(true_v: ArrayLike, false_v: ArrayLike) -> Array:
                return jnp.where(self.primal_flag(), true_v, false_v)

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

    def _or_idx(self, first: Flag, second: Flag):
        """Converts a pair of flag arrays into an array of indices for selecting between two values.

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
            An Array of indices (-1, 0, or 1) indicating which value to select from each side.
        """
        # Note that the validation has already run to check that these flags have the same shape.
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
                return Mask(chosen, FlagOp.xor_(self_flag, other_flag))

    def __invert__(self) -> "Mask[R]":
        not_flag = jtu.tree_map(FlagOp.not_, self.flag)
        return Mask(self.value, not_flag)

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


_full_slice = slice(None, None, None)

###############
# Choice maps #
###############


def _drop_prefix(
    dynamic_components: list[DynamicAddressComponent],
) -> list[DynamicAddressComponent]:
    # Check for prefix of int or scalar Array instances
    prefix_end = 0
    for comp in dynamic_components:
        if isinstance(comp, int) or (isinstance(comp, Array) and comp.shape == ()):
            prefix_end += 1
        else:
            break

    return dynamic_components[prefix_end:]


def _validate_addr(
    addr: tuple[AddressComponent, ...], allow_partial_slice: bool = False
) -> tuple[AddressComponent, ...]:
    """
    Validates the structure of an address tuple.

    This function checks if the given address adheres to the following structure:

    1. A prefix consisting of only scalar addresses (int or an IntArray with shape == ())
    2. Optionally (if `allow_partial_slice` is True), a single non-full-slice or non-scalar array
    3. A tail of full slices (: or slice(None,None,None))

    Args:
        addr: The address (or address component) to validate.
        allow_partial_slice: If True, allows a single partial slice or non-scalar array. Defaults to False.

    Returns:
        The validated address tuple.

    Raises:
        ValueError: If the address structure is invalid.
    """
    dynamic_components = [
        comp for comp in addr if isinstance(comp, (slice, int, Array))
    ]

    if dynamic_components:
        remaining = _drop_prefix(dynamic_components)

        if len(remaining) > 0:
            first = remaining[0]
            if isinstance(first, Array) and first.shape != ():
                remaining = remaining[1:]
            elif (
                allow_partial_slice
                and isinstance(first, slice)
                and first != _full_slice
            ):
                remaining = remaining[1:]

        if not all(s == _full_slice for s in remaining):
            if allow_partial_slice:
                caveat = "an optional partial slice or Array, and then only full slices"
            else:
                caveat = "full slices"

            raise ValueError(
                f"Address must consist of scalar components, followed by {caveat}. Found: {dynamic_components}"
            )

    return addr


class _SparseBuilder:
    choice_map: "Sparse | None"
    addrs: list[AddressComponent]

    def __init__(self, choice_map: "Sparse | None", addrs: list[AddressComponent]):
        self.choice_map = choice_map
        self.addrs = addrs

    def __getitem__(self, addr: Address) -> "_SparseBuilder":
        addr = addr if isinstance(addr, tuple) else (addr,)
        return _SparseBuilder(
            self.choice_map,
            [*self.addrs, *addr],
        )

    def set(self, v) -> "Sparse":
        addrs = _validate_addr(tuple(self.addrs), allow_partial_slice=False)
        chm = Sparse.entry(v, *addrs)
        if self.choice_map is None:
            return chm
        else:
            return chm + self.choice_map

    def update(self, f: Callable[..., "dict[K_addr, Any] | Sparse | Any"]) -> "Sparse":
        """
        Updates an existing value or ChoiceMap at the current address.
        This method allows updating a value or ChoiceMap at the address specified by the builder.
        The provided function `f` is called with the current value or ChoiceMap at that address.
        Args:
            f: A callable that takes the current value (or None) and returns a new value,
               dict, or ChoiceMap to be set at the current address.
        Returns:
            A new ChoiceMap with the updated value at the specified address.
        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMap.d({"x": 5, "y": {"z": 10}})
            updated = chm.at["y", "z"].update(lambda v: v * 2)
            assert updated["y", "z"] == 20
            # Updating a non-existent address
            new_chm = chm.at["w"].update(lambda _: 42)
            assert new_chm["w"] == 42
            ```
        """
        if self.choice_map is None:
            return self.set(f(_empty))
        else:
            submap = self.choice_map(tuple(self.addrs))
            if submap.has_value():
                return self.set(f(submap.get_value()))
            else:
                return self.set(f(submap))


class Sparse(Pytree):
    """The type `ChoiceMap` denotes a map-like value which can be sampled from
    generative functions.

    Generative functions which utilize `ChoiceMap` as their sample representation typically support a notion of _addressing_ for the random choices they make. `ChoiceMap` stores addressed random choices, and provides a data language for querying and manipulating these choices.

    Examples:
        (**Making choice maps**) Choice maps can be constructed using the `ChoiceMapBuilder` interface
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x"].set(3.0)
        print(chm.render_html())
        ```

        (**Getting submaps**) Hierarchical choice maps support `__call__`, which allows for the retrieval of _submaps_ at addresses:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0)
        submap = chm("x")
        print(submap.render_html())
        ```

        (**Getting values**) Choice maps support `__getitem__`, which allows for the retrieval of _values_ at addresses:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0)
        value = chm["x", "y"]
        print(value)
        ```

        (**Making vectorized choice maps**) Choice maps can be constructed using `jax.vmap`:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C
        from jax import vmap
        import jax.numpy as jnp

        vec_chm = vmap(lambda idx, v: C["x", idx].set(v))(jnp.arange(10), jnp.ones(10))
        print(vec_chm.render_html())
        ```
    """

    #######################
    # Map-like interfaces #
    #######################

    @abstractmethod
    def filter(self, selection: Selection | Flag) -> "Sparse":
        """
        Filter the choice map on the `Selection`. The resulting choice map only contains the addresses that return True when presented to the selection.

        Args:
            selection: The Selection to filter the choice map with.

        Returns:
            A new ChoiceMap containing only the addresses selected by the given Selection.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            import jax
            import genjax
            from genjax import bernoulli
            from genjax import SelectionBuilder as S


            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x


            key = jax.random.key(314159)
            tr = model.simulate(key, ())
            chm = tr.get_choices()
            selection = S["x"]
            filtered = chm.filter(selection)
            assert "y" not in filtered
            ```
        """

    @abstractmethod
    def get_value(self) -> Any:
        pass

    @abstractmethod
    def get_inner_map(
        self,
        addr: AddressComponent,
    ) -> "Sparse":
        pass

    def get_submap(self, *addresses: Address) -> "Sparse":
        addr = tuple(
            label for a in addresses for label in (a if isinstance(a, tuple) else (a,))
        )
        addr: tuple[AddressComponent, ...] = _validate_addr(
            addr, allow_partial_slice=True
        )
        return functools.reduce(lambda chm, addr: chm.get_inner_map(addr), addr, self)

    def has_value(self) -> bool:
        return self.get_value() is not None

    ######################################
    # Convenient syntax for construction #
    ######################################

    builder: Final[_SparseBuilder] = _SparseBuilder(None, [])

    ######################
    # Combinator methods #
    ######################

    def mask(self, flag: Flag) -> "Sparse":
        """
        Returns a new ChoiceMap with values masked by a boolean flag.

        This method creates a new ChoiceMap where the values are conditionally
        included based on the provided flag. If the flag is True, the original
        values are retained; if False, the ChoiceMap behaves as if it's empty.

        Args:
            flag: A boolean flag determining whether to include the values.

        Returns:
            A new ChoiceMap with values conditionally masked.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            original_chm = ChoiceMap.value(42)
            masked_chm = original_chm.mask(True)
            assert masked_chm.get_value() == 42

            masked_chm = original_chm.mask(False)
            assert masked_chm.get_value() is None
            ```
        """
        return self.filter(flag)

    def extend(self, *addrs: AddressComponent) -> "Sparse":
        """
        Returns a new ChoiceMap with the given address component as its root.

        This method creates a new ChoiceMap where the current ChoiceMap becomes a submap
        under the specified address component. It effectively adds a new level of hierarchy
        to the ChoiceMap structure.

        Args:
            addrs: The address components to use as the new root.

        Returns:
            A new ChoiceMap with the current ChoiceMap nested under the given address.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            original_chm = ChoiceMap.value(42)
            indexed_chm = original_chm.extend("x")
            assert indexed_chm["x"] == 42
            ```
        """
        acc = self
        for addr in reversed(addrs):
            if isinstance(addr, StaticAddressComponent):
                acc = Static.build({addr: acc})
            else:
                acc = Indexed.build(acc, addr)

        return acc

    def merge(self, other: "Sparse") -> "Sparse":
        """
        Merges this ChoiceMap with another ChoiceMap.

        This method combines the current ChoiceMap with another ChoiceMap using the XOR operation (^). It creates a new ChoiceMap that contains all addresses from both input ChoiceMaps; any overlapping addresses will trigger an error on access at the address via `[<addr>]` or `get_value()`. Use `|` if you don't want this behavior.

        Args:
            other: The ChoiceMap to merge with the current one.

        Returns:
            A new ChoiceMap resulting from the merge operation.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm1 = ChoiceMap.value(5).extend("x")
            chm2 = ChoiceMap.value(10).extend("y")
            merged_chm = chm1.merge(chm2)
            assert merged_chm["x"] == 5
            assert merged_chm["y"] == 10
            ```

        Note:
            This method is equivalent to using the | operator between two ChoiceMaps.
        """
        return self | other

    ###########
    # Dunders #
    ###########

    def __or__(self, other: "Sparse") -> "Sparse":
        return Or.build(self, other)

    def __add__(self, other: "Sparse") -> "Sparse":
        return self | other

    def __call__(
        self,
        *addresses: Address,
    ) -> "Sparse":
        """Alias for `get_submap(*addresses)`."""
        return self.get_submap(*addresses)

    def __getitem__(
        self,
        addr: Address,
    ):
        submap = self.get_submap(addr)
        v = submap.get_value()
        if v is None:
            raise ChoiceMapNoValueAtAddress(addr)
        else:
            return v

    def __contains__(
        self,
        addr: Address,
    ) -> bool:
        return self.get_submap(addr).has_value()

    @property
    def at(self) -> _SparseBuilder:
        """
        Returns a _ChoiceMapBuilder instance for constructing nested ChoiceMaps.

        This property allows for a fluent interface to build complex ChoiceMaps
        by chaining address components and setting values.

        Returns:
            A builder object for constructing ChoiceMaps.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            from genjax import ChoiceMap

            chm = ChoiceMap.d({("x", "y"): 3.0, "z": 12.0})
            updated = chm.at["x", "y"].set(4.0)

            assert updated["x", "y"] == 4.0
            assert updated["z"] == chm["z"]
            ```
        """
        return _SparseBuilder(self, [])


@Pytree.dataclass(match_args=True)
class Indexed(Sparse):
    """Represents a choice map with dynamic indexing.

    This class represents a choice map that uses dynamic (array-based) addressing.
    It allows for indexing into the choice map using array-like address components.

    Attributes:
        c: The underlying choice map.
        addr: The dynamic address component used for indexing.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        import jax.numpy as jnp

        base_chm = ChoiceMap.value(jnp.array([1, 2, 3]))
        idx_chm = base_chm.extend(jnp.array([0, 1, 2]))

        assert idx_chm.get_submap(1).get_value() == genjax.Mask(2, True)
        ```
    """

    c: Sparse
    addr: int | IntArray

    @staticmethod
    def build(chm: Sparse, addr: DynamicAddressComponent) -> Sparse:
        if isinstance(addr, slice):
            if addr == _full_slice:
                return chm
            else:
                raise ValueError(f"Partial slices not supported: {addr}")

        elif isinstance(addr, Array) and addr.shape == (0,):
            return Sparse.empty()

        else:
            return Indexed(chm, addr)

    def filter(self, selection: Selection | Flag) -> Sparse:
        addr = _full_slice if self.addr is None else self.addr
        return self.c.filter(selection).extend(addr)

    def get_value(self) -> Any:
        return None

    def get_inner_map(self, addr: AddressComponent) -> Sparse:
        if isinstance(addr, StaticAddressComponent):
            return Sparse.empty()

        else:
            if not isinstance(addr, slice):
                # If we allowed non-scalar addresses, the `get_submap` call would not reduce the leaf by a dimension, and further get_submap calls would target the same dimension.
                assert not jnp.asarray(addr, copy=False).shape, (
                    "Only scalar dynamic addresses are supported by get_submap."
                )

            if self.addr is None:
                # None means that this instance was created with `:`, so no masking is required and we assume that the user will provide an in-bounds `int | ScalarInt`` address. If they don't they will run up against JAX's clamping behavior.
                return jtu.tree_map(
                    lambda v: v[addr], self.c, is_leaf=lambda x: isinstance(x, Mask)
                )

            elif isinstance(self.addr, Array) and self.addr.shape:
                # We can't allow slices, as self.addr might look like, e.g. `[2,5,6]`, and we don't have any way to combine this "sparse array selector" with an incoming slice.
                assert not isinstance(addr, slice), (
                    f"Slices are not allowed against array-shaped dynamic addresses. Tried to apply {addr} to {self.addr}."
                )

                check = self.addr == addr

                # If `check` contains a match (we know it will be a single match, since we constrain addr to be scalar), then `idx` is the index of the match in `self.addr`.
                # Else, idx == 0 (selecting "junk data" of the right shape at the leaf) and check_array[idx] == False (masking the junk data).
                idx = jnp.argwhere(check, size=1, fill_value=0)[0, 0]

                return jtu.tree_map(
                    lambda v: Mask.build(v[idx], check[idx]),
                    self.c,
                    is_leaf=lambda x: isinstance(x, Mask),
                )

            else:
                return self.c.mask(self.addr == addr)


@Pytree.dataclass(match_args=True)
class Or(Sparse):
    """Represents a choice map that combines two choice maps using an OR operation.

    This class combines two choice maps, prioritizing the first choice map (c1) over the second (c2)
    when there are overlapping addresses. It returns values from c1 if present, otherwise from c2.

    Attributes:
        c1: The first choice map (higher priority).
        c2: The second choice map (lower priority).

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        chm1 = ChoiceMap.value(5)
        chm2 = ChoiceMap.value(10)
        or_chm = chm1 | chm2
        assert or_chm.get_value() == 5  # c1 takes priority

        chm3 = ChoiceMap.empty()
        chm4 = ChoiceMap.value(15)
        or_chm2 = chm3 | chm4
        assert or_chm2.get_value() == 15  # c2 used when c1 is empty
        ```
    """

    c1: Sparse
    c2: Sparse

    @staticmethod
    def build(
        c1: Sparse,
        c2: Sparse,
    ) -> Sparse:
        if c2.static_is_empty():
            return c1
        elif c1.static_is_empty():
            return c2
        else:
            return Or(c1, c2)

    def filter(self, selection: Selection | Flag) -> Sparse:
        return self.c1.filter(selection) | self.c2.filter(selection)

    def get_value(self) -> Any:
        return None

    def get_inner_map(self, addr: AddressComponent) -> Sparse:
        submap1 = self.c1.get_inner_map(addr)
        submap2 = self.c2.get_inner_map(addr)
        return submap1 | submap2
