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

from abc import abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import or_

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative.core import Constraint, ProjectProblem, Sample
from genjax._src.core.generative.functional_types import Mask, Sum
from genjax._src.core.interpreters.staging import (
    Flag,
    staged_err,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Bool,
    BoolArray,
    EllipsisType,
    Generic,
    String,
    TypeVar,
)

T = TypeVar("T")

#################
# Address types #
#################

StaticAddressComponent = String
DynamicAddressComponent = ArrayLike
AddressComponent = StaticAddressComponent | DynamicAddressComponent
Address = tuple[()] | tuple[AddressComponent, ...]
StaticAddress = tuple[()] | tuple[StaticAddressComponent, ...]
ExtendedStaticAddressComponent = StaticAddressComponent | EllipsisType
ExtendedAddressComponent = ExtendedStaticAddressComponent | DynamicAddressComponent
ExtendedAddress = tuple[()] | tuple[ExtendedAddressComponent, ...]

##############
# Selections #
##############


###############################
# Selection builder interface #
###############################


@Pytree.dataclass
class _SelectionBuilder(Pytree):
    def __getitem__(self, addr_comps):
        if not isinstance(addr_comps, tuple):
            addr_comps = (addr_comps,)

        sel = Selection.all()
        for comp in reversed(addr_comps):
            sel = sel.indexed(comp)
        return sel


SelectionBuilder = _SelectionBuilder()


class Selection(ProjectProblem):
    """The type `Selection` provides a lens-like interface for filtering the
    random choices in a `ChoiceMap`.

    Examples:
        (**Making selections**) Selections can be constructed using the `SelectionBuilder` interface
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import SelectionBuilder as S

        sel = S["x", "y"]
        print(sel.render_html())
        ```

        (**Getting subselections**) Hierarchical selections support `__call__`, which allows for the retrieval of _subselections_ at addresses:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel = S["x", "y"]
        subsel = sel("x")
        print(subsel.render_html())
        ```

        (**Check for inclusion**) Selections support `__getitem__`, which provides a way to check if an address is included in the selection:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel = S["x", "y"]
        not_included = sel["x"]
        included = sel["x", "y"]
        print(not_included, included)
        ```

        (**Complement selections**) Selections can be complemented:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel = ~S["x", "y"]
        included = sel["x"]
        not_included = sel["x", "y"]
        print(included, not_included)
        ```

        (**Combining selections**) Selections can be combined, via the `|` syntax:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel = S["x", "y"] | S["z"]
        print(sel["x", "y"], sel["z", "y"])
        ```
    """

    #################################################
    # Convenient syntax for constructing selections #
    #################################################

    @classmethod
    def all(cls) -> "Selection":
        """
        Returns a Selection that selects all addresses.

        This method creates and returns an instance of AllSel, which represents
        a selection that includes all possible addresses in a ChoiceMap.

        Returns:
            A Selection that selects everything.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            all_selection = Selection.all()
            assert all_selection["any_address"] == True
            ```
        """
        return AllSel()

    @classmethod
    def none(cls) -> "Selection":
        """
        Returns a Selection that selects no addresses.

        This method creates and returns an instance of ComplementSel(AllSel()),
        which represents a selection that excludes all possible addresses in a ChoiceMap.

        Returns:
            A Selection that selects nothing.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            none_selection = Selection.none()
            assert none_selection["any_address"] == False
            ```
        """
        return ~cls.all()

    ######################
    # Combinator methods #
    ######################

    def __or__(self, other: "Selection") -> "Selection":
        return OrSel(self, other)

    def __and__(self, other: "Selection") -> "Selection":
        return AndSel(self, other)

    def __invert__(self) -> "Selection":
        return ComplementSel(self)

    def maybe(self, flag: Flag) -> "Selection":
        """
        Returns a new Selection that is conditionally applied based on a flag.

        This method creates a new Selection that applies the current selection
        only if the given flag is True. If the flag is False, the resulting
        selection will not select any addresses.

        Args:
            flag: A boolean flag determining whether the selection is applied.

        Returns:
            A new Selection that is conditionally applied based on the flag.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            base_selection = Selection.all()
            maybe_selection = base_selection.maybe(Flag(True))
            assert maybe_selection["any_address"] == True

            maybe_selection = base_selection.maybe(Flag(False))
            assert maybe_selection["any_address"] == False
            ```
        """
        return DeferSel(self, flag)

    def indexed(self, addr: ExtendedAddressComponent) -> "Selection":
        """
        Returns a new Selection that is indexed by the given address component.

        This method creates a new Selection that applies the current selection
        to the specified address component. It handles both static and dynamic
        address components.

        Args:
            addr: The address component to index the selection.

        Returns:
            A new Selection indexed by the given address component.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            base_selection = Selection.all()
            indexed_selection = base_selection.indexed("x")
            assert indexed_selection["x"]["any_subaddress"] == True
            assert indexed_selection["y"].check() == False
            ```
        """
        if isinstance(addr, ExtendedStaticAddressComponent):
            return StaticSel(self, addr)
        else:
            return IdxSel(self, addr)

    def __call__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ) -> "Selection":
        addr = addr if isinstance(addr, tuple) else (addr,)
        subselection = self
        for comp in addr:
            subselection = subselection.get_subselection(comp)
        return subselection

    def __getitem__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ) -> Flag:
        subselection = self(addr)
        return subselection.check()

    def __contains__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ) -> Flag:
        return self[addr]

    @abstractmethod
    def check(self) -> Flag:
        raise NotImplementedError

    @abstractmethod
    def get_subselection(self, addr: ExtendedAddressComponent) -> "Selection":
        raise NotImplementedError


#######################
# Selection functions #
#######################


@Pytree.dataclass
class AllSel(Selection):
    """Represents a selection that includes all addresses.

    This selection always returns True for any address check and returns itself
    for any subselection, effectively representing a selection of all possible
    addresses in a choice map.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        all_sel = Selection.all()
        assert all_sel.check() == True
        assert all_sel["any_address"] == True
        assert isinstance(all_sel("any_address"), AllSel)
        ```
    """

    def check(self) -> Flag:
        return Flag(True)

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        return AllSel()


@Pytree.dataclass
class DeferSel(Selection):
    """Represents a deferred selection that is conditionally applied based on a flag.

    This selection wraps another selection and a boolean flag. The check and
    subselection operations are only applied if the flag is True. If the flag
    is False, the selection behaves as if it's empty.

    Attributes:
        s: The underlying selection to be conditionally applied.
        flag: A boolean flag determining whether the selection is active.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        base_sel = Selection.all()
        flag = Flag(True)
        defer_sel = base_sel.maybe(flag)
        assert defer_sel.check() == True

        flag = Flag(False)
        defer_sel = base_sel.maybe(flag)
        assert defer_sel.check() == False
        ```
    """

    s: Selection
    flag: Flag

    def check(self) -> Flag:
        ch = self.s.check()
        return self.flag.and_(ch)

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining = self.s(addr)
        return remaining.maybe(self.flag)


@Pytree.dataclass
class ComplementSel(Selection):
    """Represents the complement of a selection.

    This selection inverts the behavior of another selection. It checks for the
    opposite of what the wrapped selection checks for, and returns the complement
    of its subselections.

    Attributes:
        s: The selection to be complemented.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        base_sel = Selection.all()
        comp_sel = ~base_sel
        assert comp_sel.check() == False

        specific_sel = SelectionBuilder["x", "y"]
        comp_specific = ~specific_sel
        assert comp_specific["x", "y"] == False
        assert comp_specific["z"] == True
        ```
    """

    s: Selection

    def check(self) -> Flag:
        return self.s.check().not_()

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining = self.s(addr)
        return ~remaining


@Pytree.dataclass
class StaticSel(Selection):
    """Represents a static selection based on a specific address component.

    This selection is used to filter choices based on a static address component.
    It always returns False for the check method, as it's meant to be used in
    combination with other selections or as part of a larger selection structure.

    Attributes:
        s: The underlying selection to be applied if the address matches.
        addr: The static address component to match against.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        static_sel = Selection.all().indexed("x")
        assert static_sel.check() == False
        assert static_sel.get_subselection("x").check() == True
        assert static_sel.get_subselection("y").check() == False
        ```
    """

    s: Selection = Pytree.field()
    addr: ExtendedStaticAddressComponent = Pytree.static()

    def check(self) -> Flag:
        return Flag(False)

    def get_subselection(self, addr: EllipsisType | AddressComponent) -> Selection:
        check = Flag(addr == self.addr or isinstance(addr, EllipsisType))
        return self.s.maybe(check)


@Pytree.dataclass
class IdxSel(Selection):
    """Represents a dynamic selection based on an array of address components.

    This selection is used to filter choices based on dynamic address components.
    It always returns False for the check method, as it's meant to be used in
    combination with other selections or as part of a larger selection structure.

    Attributes:
        s: The underlying selection to be applied if the address matches.
        idxs: The dynamic address components to match against.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        import jax.numpy as jnp

        idx_sel = Selection.all().indexed(jnp.array([1, 2, 3]))
        assert idx_sel.check() == False
        assert idx_sel.get_subselection(2).check() == True
        assert idx_sel.get_subselection(4).check() == False
        ```
    """

    s: Selection
    idxs: DynamicAddressComponent

    def check(self) -> Flag:
        return Flag(False)

    def get_subselection(self, addr: EllipsisType | AddressComponent) -> Selection:
        if isinstance(addr, EllipsisType):
            return self.s

        if not isinstance(addr, DynamicAddressComponent):
            return Selection.none()

        else:

            def check_fn(v):
                return jnp.logical_and(
                    v,
                    jnp.any(v == self.idxs),
                )

            check = Flag(
                jax.vmap(check_fn)(addr)
                if jnp.array(addr, copy=False).shape
                else check_fn(addr)
            )
            return self.s.maybe(check)


@Pytree.dataclass
class AndSel(Selection):
    """Represents a selection that combines two other selections using a logical AND operation.

    This selection is true only if both of its constituent selections are true.
    It allows for the combination of multiple selection criteria.

    Attributes:
        s1: The first selection to be combined.
        s2: The second selection to be combined.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel1 = Selection.all().indexed("x")
        sel2 = Selection.all().indexed("y")
        and_sel = sel1 & sel2
        assert and_sel["x", "y"] == True
        assert and_sel["x"] == False
        assert and_sel["y"] == False
        ```
    """

    s1: Selection
    s2: Selection

    def check(self) -> Flag:
        return self.s1.check().and_(self.s2.check())

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return remaining1 & remaining2


@Pytree.dataclass
class OrSel(Selection):
    """Represents a selection that combines two other selections using a logical OR operation.

    This selection is true if either of its constituent selections is true.
    It allows for the combination of multiple selection criteria using an inclusive OR.

    Attributes:
        s1: The first selection to be combined.
        s2: The second selection to be combined.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel1 = Selection.all().indexed("x")
        sel2 = Selection.all().indexed("y")
        or_sel = sel1 | sel2
        assert or_sel["x", "y"] == True
        assert or_sel["x"] == True
        assert or_sel["y"] == True
        assert or_sel["z"] == False
        ```
    """

    s1: Selection
    s2: Selection

    def check(self) -> Flag:
        return self.s1.check().or_(self.s2.check())

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return remaining1 | remaining2


@Pytree.dataclass
class ChmSel(Selection):
    """Represents a selection based on a ChoiceMap.

    This selection is true for addresses that have a value in the associated ChoiceMap.
    It allows for creating selections that match the structure of existing ChoiceMaps.

    Attributes:
        c: The ChoiceMap on which this selection is based.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0) ^ C["z"].set(5.0)
        sel = chm.get_selection()
        assert sel["x", "y"] == True
        assert sel["z"] == True
        assert sel["w"] == False
        ```
    """

    c: "ChoiceMap"

    def check(self) -> Flag:
        return check_none(self.c.get_value())

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        submap = self.c.get_submap(addr)
        return submap.get_selection()


###############
# Choice maps #
###############


@dataclass
class ChoiceMapNoValueAtAddress(Exception):
    subaddr: ExtendedAddressComponent | ExtendedAddress


@Pytree.dataclass
class _ChoiceMapBuilder(Pytree):
    addr: ExtendedAddress = ()

    def __getitem__(
        self, addr: ExtendedAddressComponent | ExtendedAddress
    ) -> "_ChoiceMapBuilder":
        addr = addr if isinstance(addr, tuple) else (addr,)
        return _ChoiceMapBuilder(
            addr,
        )

    def set(self, v) -> "ChoiceMap":
        """
        Creates a ChoiceMap with a value at the specified address.

        This method constructs a ChoiceMap by setting a value at the address
        specified in the builder. If no address is specified, it creates a
        ChoiceMap with just the value.

        Args:
            v: The value to be stored in the ChoiceMap.

        Returns:
            A new ChoiceMap containing the value at the specified address.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMapBuilder["x", "y"].set(42)
            assert chm["x", "y"] == 42
            ```
        """
        if self.addr:
            return self.a(self.addr, v)
        else:
            return _empty

    def n(self) -> "ChoiceMap":
        """
        Creates an empty ChoiceMap.

        This method constructs and returns an empty ChoiceMap. It's a convenient
        way to create a ChoiceMap with no entries, which can be useful as a starting
        point for building more complex ChoiceMaps or when you need to represent
        the absence of choices.

        Returns:
            An empty ChoiceMap.
            ```
        """
        return _empty

    def v(self, v) -> "ChoiceMap":
        """
        Creates a ChoiceMap with a single value.

        This method constructs a ChoiceMap containing a single value. It's a convenient
        way to create a ChoiceMap with a single entry without specifying an address.

        Args:
            v: The value to be stored in the ChoiceMap.

        Returns:
            A new ChoiceMap containing the single value.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMapBuilder.v(42)
            assert chm.get_value() == 42
            ```
        """
        return ChoiceMap.value(v)

    def d(self, d: dict[Any, Any]) -> "ChoiceMap":
        """
        Creates a ChoiceMap from a dictionary.

        This method constructs a ChoiceMap by iterating through the key-value pairs
        of the provided dictionary. Each key-value pair is converted into a ChoiceMap
        entry, where the key becomes the address and the value becomes the choice.

        Args:
            d: A dictionary where keys are addresses and values are choices.

        Returns:
            A new ChoiceMap containing the entries from the dictionary.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMapBuilder.d({"x": 1, "y": 2})
            assert chm["x"].get_value() == 1
            assert chm["y"].get_value() == 2
            ```
        """
        return ChoiceMap.d(d)

    def kw(self, **kwargs) -> "ChoiceMap":
        """
        Creates a ChoiceMap from keyword arguments.

        This method constructs a ChoiceMap by using the keyword arguments provided.
        Each keyword argument becomes an entry in the ChoiceMap, where the argument
        name is the address and the argument value is the choice.

        Args:
            **kwargs: Arbitrary keyword arguments where keys are addresses and values are choices.

        Returns:
            A new ChoiceMap containing the entries from the keyword arguments.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMapBuilder.kw(x=1, y=2)
            assert chm["x"].get_value() == 1
            assert chm["y"].get_value() == 2
            ```
        """
        return ChoiceMap.kw(**kwargs)

    def a(
        self, addr: ExtendedAddressComponent | ExtendedAddress, v: Any
    ) -> "ChoiceMap":
        """
        Creates a ChoiceMap with a single address-value pair.

        This method constructs a ChoiceMap containing a single entry with the specified
        address and value. If the value is already a ChoiceMap, it is used directly;
        otherwise, it is wrapped in a ChoiceMap.

        Args:
            addr: The address for the choice. Can be a single address component or a tuple of components.
            v: The value to associate with the address. If it's not a ChoiceMap, it will be wrapped in one.

        Returns:
            A new ChoiceMap containing the single address-value pair.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMapBuilder.a("x", 1)
            assert chm["x"].get_value() == 1

            nested_chm = ChoiceMapBuilder.a(("x", "y"), 2)
            assert nested_chm["x"]["y"].get_value() == 2
            ```
        """
        addr = addr if isinstance(addr, tuple) else (addr,)
        new = ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        for comp in reversed(addr):
            new = ChoiceMap.idx(new, comp)

        return new


ChoiceMapBuilder = _ChoiceMapBuilder()


def check_none(v: Any | Mask[Any] | None) -> Flag:
    if v is None:
        return Flag(False)
    elif isinstance(v, Mask):
        return v.flag
    else:
        return Flag(True)


##########################
# AddressIndex interface #
##########################


@Pytree.dataclass
class AddressIndex(Pytree):
    choice_map: "ChoiceMap"
    addrs: list[Address]

    def __getitem__(self, addr: AddressComponent | Address) -> "AddressIndex":
        addr = addr if isinstance(addr, tuple) else (addr,)
        return AddressIndex(
            self.choice_map,
            [*self.addrs, addr],
        )

    def set(self, v):
        new = self.choice_map
        for addr in self.addrs:
            new = ChoiceMapBuilder.a(addr, v) + new
        return new

    @property
    def at(self) -> "AddressIndex":
        return self

    def filter(self):
        sels = map(lambda addr: SelectionBuilder[addr], self.addrs)
        or_sel = reduce(or_, sels)
        return self.choice_map.filter(or_sel)


class ChoiceMap(Sample, Constraint):
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
    def get_value(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_submap(
        self,
        addr: ExtendedAddressComponent,
    ) -> "ChoiceMap":
        raise NotImplementedError

    def has_value(self) -> Flag:
        return check_none(self.get_value())

    ######################################
    # Convenient syntax for construction #
    ######################################

    @classmethod
    def empty(cls) -> "EmptyChm":
        """
        Returns an empty ChoiceMap.

        This method creates and returns an instance of EmptyChm, which represents
        a ChoiceMap with no values or submaps.

        Returns:
            An empty ChoiceMap.
        """
        return _empty

    @classmethod
    def value(cls, v: T) -> "ValueChm[T]":
        """
        Creates a ChoiceMap containing a single value.

        This method creates and returns an instance of ValueChm, which represents
        a ChoiceMap with a single value at the root level.

        Args:
            v: The value to be stored in the ChoiceMap.

        Returns:
            A ChoiceMap containing the single value.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            value_chm = ChoiceMap.value(42)
            assert value_chm.get_value() == 42
            ```
        """
        return ValueChm(v)

    @classmethod
    def idx(cls, v: Any, addr: ExtendedAddressComponent) -> "ChoiceMap":
        """
        Creates a ChoiceMap with a single value at a specified address.

        This method creates and returns a ChoiceMap with a single value stored at
        the given address. If the provided value is already a ChoiceMap, it will
        be used directly; otherwise, it will be wrapped in a ValueChm.

        Args:
            v: The value to be stored in the ChoiceMap. Can be any value or a ChoiceMap.
            addr: The address at which to store the value. Can be a static or dynamic address component.

        Returns:
            A ChoiceMap with the value stored at the specified address.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            # Static address
            static_chm = ChoiceMap.idx(42, "x")
            assert static_chm["x"].get_value() == 42

            # Dynamic address
            dynamic_chm = ChoiceMap.idx(42, jnp.array([1, 2, 3]))
            assert dynamic_chm[jnp.array([1, 2, 3])].get_value() == 42

            # Using an existing ChoiceMap
            nested_chm = ChoiceMap.idx(ChoiceMap.value(42), "x")
            assert nested_chm["x"].get_value() == 42
            ```
        """
        chm = v if isinstance(v, ChoiceMap) else ChoiceMap.value(v)
        if isinstance(addr, ExtendedStaticAddressComponent):
            return StaticChm.build(chm, addr)
        else:
            return IdxChm.build(chm, addr)

    @classmethod
    def d(cls, d: dict[Any, Any]) -> "ChoiceMap":
        """
        Creates a ChoiceMap from a dictionary.

        This method creates and returns a ChoiceMap based on the key-value pairs
        in the provided dictionary. Each key in the dictionary becomes an address
        in the ChoiceMap, and the corresponding value is stored at that address.

        Args:
            d: A dictionary where keys are addresses and values are the corresponding
               data to be stored in the ChoiceMap.

        Returns:
            A ChoiceMap containing the key-value pairs from the input dictionary.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            dict_chm = ChoiceMap.d({"x": 42, "y": [1, 2, 3]})
            assert dict_chm["x"].get_value() == 42
            assert dict_chm["y"].get_value() == [1, 2, 3]
            ```
        """
        start = ChoiceMap.empty()
        if d:
            for k, v in d.items():
                start = ChoiceMapBuilder.a(k, v) ^ start
        return start

    @classmethod
    def kw(cls, **kwargs) -> "ChoiceMap":
        """
        Creates a ChoiceMap from keyword arguments.

        This method creates and returns a ChoiceMap based on the provided keyword arguments.
        Each keyword argument becomes an address in the ChoiceMap, and its value is stored at that address.

        Returns:
            A ChoiceMap containing the key-value pairs from the input keyword arguments.

        Example:
            ```python
            kw_chm = ChoiceMap.kw(x=42, y=[1, 2, 3])
            assert kw_chm["x"].get_value() == 42
            assert kw_chm["y"].get_value() == [1, 2, 3]
            ```
        """
        return ChoiceMap.d(kwargs)

    ######################
    # Combinator methods #
    ######################

    def filter(self, selection: Selection) -> "ChoiceMap":
        """
        Filter the choice map on the `Selection`. The resulting choice map only contains the addresses in the selection.

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


            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.get_sample()
            selection = S["x"]
            filtered = chm.filter(selection)
            print("y" in filtered)
            ```
        """
        return FilteredChm.build(self, selection)

    def indexed(self, addr: ExtendedAddressComponent) -> "ChoiceMap":
        """
        Returns a new ChoiceMap with the given address component as its root.

        This method creates a new ChoiceMap where the current ChoiceMap becomes a submap
        under the specified address component. It effectively adds a new level of hierarchy
        to the ChoiceMap structure.

        Args:
            addr: The address component to use as the new root.

        Returns:
            A new ChoiceMap with the current ChoiceMap nested under the given address.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            original_chm = ChoiceMap.value(42)
            indexed_chm = original_chm.indexed("x")
            assert indexed_chm["x"].get_value() == 42
            ```
        """
        return ChoiceMap.idx(self, addr)

    def mask(self, f: Flag) -> "ChoiceMap":
        """
        Returns a new ChoiceMap with values masked by a boolean flag.

        This method creates a new ChoiceMap where the values are conditionally
        included based on the provided flag. If the flag is True, the original
        values are retained; if False, the ChoiceMap behaves as if it's empty.

        Args:
            f: A boolean flag determining whether to include the values.

        Returns:
            A new ChoiceMap with values conditionally masked.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            original_chm = ChoiceMap.value(42)
            flag = Flag(True)
            masked_chm = original_chm.mask(flag)
            assert masked_chm.get_value() == 42

            flag = Flag(False)
            masked_chm = original_chm.mask(flag)
            assert masked_chm.get_value() is None
            ```
        """
        return MaskChm.build(self, f)

    def merge(self, other: "ChoiceMap") -> "ChoiceMap":
        """
        Merges this ChoiceMap with another ChoiceMap.

        This method combines the current ChoiceMap with another ChoiceMap using
        the XOR operation (^). It creates a new ChoiceMap that contains all
        addresses from both input ChoiceMaps, with values from the second
        ChoiceMap taking precedence in case of overlapping addresses.

        Args:
            other: The ChoiceMap to merge with the current one.

        Returns:
            A new ChoiceMap resulting from the merge operation.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm1 = ChoiceMap.value(5).indexed("x")
            chm2 = ChoiceMap.value(10).indexed("y")
            merged_chm = chm1.merge(chm2)
            assert merged_chm["x"] == 5
            assert merged_chm["y"] == 10
            ```

        Note:
            This method is equivalent to using the ^ operator between two ChoiceMaps.
        """
        return self ^ other

    def get_selection(self) -> Selection:
        """
        Returns a Selection representing the structure of this ChoiceMap.

        This method creates a Selection that matches the hierarchical structure
        of the current ChoiceMap. The resulting Selection can be used to filter
        or query other ChoiceMaps with the same structure.

        Returns:
            A Selection object representing the structure of this ChoiceMap.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMap.value(5).indexed("x")
            sel = chm.get_selection()
            assert sel["x"] == True
            assert sel["y"] == False
            ```
        """
        return ChmSel(self)

    def static_is_empty(self) -> Bool:
        return False

    ###########
    # Dunders #
    ###########

    def __xor__(self, other: "ChoiceMap") -> "ChoiceMap":
        return XorChm.build(self, other)

    def __add__(self, other: "ChoiceMap") -> "ChoiceMap":
        return OrChm.build(self, other)

    def __or__(self, other):
        return OrChm.build(self, other)

    def __call__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ):
        addr = addr if isinstance(addr, tuple) else (addr,)
        submap = self
        for comp in addr:
            submap = submap.get_submap(comp)
        return submap

    def __getitem__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ):
        addr = addr if isinstance(addr, tuple) else (addr,)
        submap = self(addr)
        v = submap.get_value()
        if v is None:
            raise ChoiceMapNoValueAtAddress(addr)
        else:
            return v

    def __contains__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ):
        addr = addr if isinstance(addr, tuple) else (addr,)
        submap = self
        for comp in addr:
            submap = self.get_submap(comp)
        return submap.has_value()

    @property
    def at(self) -> AddressIndex:
        """Access the `ChoiceMap.AddressIndex` mutation interface. This allows users to take an existing choice map, and mutate it _functionally_.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = C["x", "y"].set(3.0)
            chm = chm.at["x", "y"].set(4.0)
            print(chm["x", "y"])
            ```
        """
        return AddressIndex(self, [])


@Pytree.dataclass
class EmptyChm(ChoiceMap):
    """Represents an empty choice map.

    This class represents a choice map with no values or submaps. It serves as the base
    case for hierarchical choice maps and is used when no choices have been made or
    when all choices have been filtered out.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        empty_chm = ChoiceMap.empty()
        assert empty_chm.get_value() is None
        assert empty_chm.get_submap("any_address") == empty_chm
        assert empty_chm.static_is_empty() == True
        ```
    """

    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        return self

    def static_is_empty(self) -> Bool:
        return True


_empty = EmptyChm()


@Pytree.dataclass
class ValueChm(Generic[T], ChoiceMap):
    """Represents a choice map with a single value.

    This class represents a choice map that contains a single value at the root level.
    It is used to store individual choices in a hierarchical choice map structure.

    Attributes:
        v: The value stored in this choice map.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        value_chm = ChoiceMap.value(3.14)
        assert value_chm.get_value() == 3.14
        assert value_chm.get_submap("any_address").static_is_empty() == True
        ```
    """

    v: T

    def get_value(self) -> T:
        return self.v

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        return _empty


@Pytree.dataclass
class IdxChm(ChoiceMap):
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
        idx_chm = base_chm.indexed(jnp.array([0, 1, 2])
        assert idx_chm.get_submap(1).get_value() == 2
        assert idx_chm.get_submap(3).static_is_empty() == True
        ```
    """

    c: ChoiceMap
    addr: DynamicAddressComponent

    @classmethod
    def build(cls, chm: ChoiceMap, addr: DynamicAddressComponent) -> ChoiceMap:
        return _empty if chm.static_is_empty() else IdxChm(chm, addr)

    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        if addr is Ellipsis:
            return self.c

        elif not isinstance(addr, DynamicAddressComponent):
            return _empty

        else:

            def check_fn(idx, addr) -> BoolArray:
                return jnp.array(idx == addr, copy=False)

            check = (
                jax.vmap(check_fn, in_axes=(None, 0))(addr, self.addr)
                if jnp.array(self.addr, copy=False).shape
                else check_fn(addr, self.addr)
            )

            return (
                MaskChm.build(
                    jtu.tree_map(lambda v: v[addr], self.c), Flag(check[addr])
                )
                if jnp.array(check, copy=False).shape
                else self.c.mask(Flag(check))
            )


@Pytree.dataclass
class StaticChm(ChoiceMap):
    """Represents a static choice map with a fixed address component.

    This class is used to create a choice map that is indexed by a static address component.
    It wraps another choice map and provides access to it only when the given address matches.

    Attributes:
        c: The underlying choice map.
        addr: The static address component used for indexing.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        base_chm = ChoiceMap.value(5)
        static_chm = base_chm.indexed("x")
        assert static_chm.get_submap("x").get_value() == 5
        assert static_chm.get_submap("y").static_is_empty() == True
        ```
    """

    c: ChoiceMap = Pytree.field()
    addr: ExtendedStaticAddressComponent = Pytree.static()

    @classmethod
    def build(
        cls,
        c: ChoiceMap,
        addr: ExtendedStaticAddressComponent,
    ) -> ChoiceMap:
        return _empty if c.static_is_empty() else StaticChm(c, addr)

    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        check = Flag(addr == self.addr)
        return self.c.mask(check)


@Pytree.dataclass
class XorChm(ChoiceMap):
    """Represents a disjoint union of two choice maps.

    This class combines two choice maps in a way that ensures their domains are disjoint.
    It's used to merge two choice maps while preventing overlapping addresses.

    Attributes:
        c1: The first choice map.
        c2: The second choice map.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        chm1 = ChoiceMap.value(5).indexed("x")
        chm2 = ChoiceMap.value(10).indexed("y")
        xor_chm = chm1 ^ chm2
        assert xor_chm.get_submap("x").get_value() == 5
        assert xor_chm.get_submap("y").get_value() == 10
        ```

    Raises:
        RuntimeError: If there's a value collision between the two choice maps.
    """

    c1: ChoiceMap
    c2: ChoiceMap

    @classmethod
    def build(
        cls,
        c1: ChoiceMap,
        c2: ChoiceMap,
    ) -> ChoiceMap:
        match (c1.static_is_empty(), c2.static_is_empty()):
            case True, True:
                return _empty
            case _, True:
                return c1
            case True, _:
                return c2
            case _:
                return XorChm(c1, c2)

    def get_value(self) -> Any:
        check1 = self.c1.has_value()
        check2 = self.c2.has_value()
        err_check = check1.and_(check2)
        staged_err(
            err_check,
            f"The disjoint union of two choice maps have a value collision:\nc1 = {self.c1}\nc2 = {self.c2}",
        )
        v1 = self.c1.get_value()
        v2 = self.c2.get_value()

        def pair_bool_to_idx(bool1, bool2):
            return 1 * bool1.f + 2 * bool2.f - 3 * bool1.and_(bool2).f - 1

        idx = pair_bool_to_idx(check1, check2)
        return Sum.maybe_none(idx, [v1, v2])

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        remaining_1 = self.c1.get_submap(addr)
        remaining_2 = self.c2.get_submap(addr)
        return remaining_1 ^ remaining_2


@Pytree.dataclass
class OrChm(ChoiceMap):
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

    c1: ChoiceMap
    c2: ChoiceMap

    @classmethod
    def build(
        cls,
        c1: ChoiceMap,
        c2: ChoiceMap,
    ) -> ChoiceMap:
        match (c1.static_is_empty(), c2.static_is_empty()):
            case True, True:
                return _empty
            case _, True:
                return c1
            case True, _:
                return c2
            case _:
                return OrChm(c1, c2)

    def get_value(self) -> Any:
        check1 = self.c1.has_value()
        check2 = self.c2.has_value()
        v1 = self.c1.get_value()
        v2 = self.c2.get_value()

        def pair_bool_to_idx(first, second):
            output = -1 + first.f + 2 * first.not_().and_(second).f
            return output

        idx = pair_bool_to_idx(check1, check2)
        return Sum.maybe_none(idx, [v1, v2])

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        submap1 = self.c1.get_submap(addr)
        submap2 = self.c2.get_submap(addr)

        return submap1 | submap2


@Pytree.dataclass
class MaskChm(ChoiceMap):
    """Represents a choice map that is conditionally masked.

    This class wraps another choice map and applies a boolean flag to conditionally
    mask its contents. When the flag is True, the underlying choice map behaves
    normally. When the flag is False, the choice map appears empty.

    Attributes:
        c: The underlying choice map.
        flag: A boolean flag determining whether the choice map is masked.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        base_chm = ChoiceMap.value(10)
        mask_chm = base_chm.mask(Flag(True))
        assert mask_chm.get_value() == 10

        mask_chm = base_chm.mask(Flag(False))
        assert mask_chm.get_value() is None
        assert mask_chm.static_is_empty() == False
        ```
    """

    c: ChoiceMap
    flag: Flag

    @classmethod
    def build(
        cls,
        c: ChoiceMap,
        flag: Flag,
    ) -> ChoiceMap:
        return (
            c
            if c.static_is_empty()
            else c
            if flag.concrete_true()
            else _empty
            if flag.concrete_false()
            else MaskChm(c, flag)
        )

    def get_value(self) -> Any:
        v = self.c.get_value()
        return Mask.maybe_none(self.flag, v)

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        return submap.mask(self.flag)


@Pytree.dataclass
class FilteredChm(ChoiceMap):
    """Represents a filtered choice map based on a selection.

    This class wraps another choice map and applies a selection to filter its contents.
    It allows for selective access to the underlying choice map based on the provided selection.

    Attributes:
        c: The underlying choice map.
        selection: The selection used to filter the choice map.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import SelectionBuilder as S

        base_chm = ChoiceMap.value(10).indexed("x")
        filtered_chm = base_chm.filter(S["x"])
        assert filtered_chm.get_submap("x").get_value() == 10

        filtered_chm = base_chm.filter(S["y"])
        assert filtered_chm.get_submap("x").static_is_empty() == True
        ```
    """

    c: ChoiceMap
    selection: Selection

    @classmethod
    def build(cls, chm: ChoiceMap, selection: Selection) -> ChoiceMap:
        return _empty if chm.static_is_empty() else FilteredChm(chm, selection)

    def get_value(self) -> Any:
        v = self.c.get_value()
        sel_check = self.selection[()]
        return Mask.maybe_none(sel_check, v)

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        subselection = self.selection(addr)
        return submap.filter(subselection)
