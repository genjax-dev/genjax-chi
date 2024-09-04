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
        ```python exec="yes" source="material-block" session="core"
        from genjax import SelectionBuilder as S

        sel = S["x", "y"]
        print(sel.render_html())
        ```

        (**Getting subselections**) Hierarchical selections support `__call__`, which allows for the retrieval of _subselections_ at addresses:
        ```python exec="yes" source="material-block" session="core"
        sel = S["x", "y"]
        subsel = sel("x")
        print(subsel.render_html())
        ```

        (**Check for inclusion**) Selections support `__getitem__`, which provides a way to check if an address is included in the selection:
        ```python exec="yes" source="material-block" session="core"
        sel = S["x", "y"]
        not_included = sel["x"]
        included = sel["x", "y"]
        print(not_included, included)
        ```

        (**Complement selections**) Selections can be complemented:
        ```python exec="yes" source="material-block" session="core"
        sel = ~S["x", "y"]
        included = sel["x"]
        not_included = sel["x", "y"]
        print(included, not_included)
        ```

        (**Combining selections**) Selections can be combined, via the `|` syntax:
        ```python exec="yes" source="material-block" session="core"
        sel = S["x", "y"] | S["z"]
        print(sel["x", "y"], sel["z", "y"])
        ```
    """

    #################################################
    # Convenient syntax for constructing selections #
    #################################################

    @classmethod
    def all(cls) -> "Selection":
        return AllSel()

    @classmethod
    def none(cls) -> "Selection":
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
        return DeferSel(self, flag)

    def indexed(self, addr: ExtendedAddressComponent) -> "Selection":
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
    def check(self) -> Flag:
        return Flag(True)

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        return AllSel()


@Pytree.dataclass
class DeferSel(Selection):
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
    s: Selection

    def check(self) -> Flag:
        return self.s.check().not_()

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining = self.s(addr)
        return ~remaining


@Pytree.dataclass
class StaticSel(Selection):
    s: Selection = Pytree.field()
    addr: ExtendedStaticAddressComponent = Pytree.static()

    def check(self) -> Flag:
        return Flag(False)

    def get_subselection(self, addr: EllipsisType | AddressComponent) -> Selection:
        check = Flag(addr == self.addr or isinstance(addr, EllipsisType))
        return self.s.maybe(check)


@Pytree.dataclass
class IdxSel(Selection):
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
    """A Selection that wraps a ChoiceMap.

    This class allows a ChoiceMap to be used as a Selection, enabling filtering and selection operations based on the structure of the ChoiceMap.

    Attributes:
        c: The wrapped ChoiceMap.
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
        if self.addr:
            return self.a(self.addr, v)
        else:
            return _empty

    def n(self) -> "ChoiceMap":
        return _empty

    def v(self, v) -> "ChoiceMap":
        return ChoiceMap.value(v)

    def d(self, d: dict[Any, Any]) -> "ChoiceMap":
        return ChoiceMap.d(d)

    def kw(self, **kwargs) -> "ChoiceMap":
        return ChoiceMap.kw(**kwargs)

    def a(
        self, addr: ExtendedAddressComponent | ExtendedAddress, v: Any
    ) -> "ChoiceMap":
        addr = addr if isinstance(addr, tuple) else (addr,)
        new = ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        for comp in reversed(addr):
            if isinstance(comp, StaticAddressComponent):
                new = StaticChm.build(comp, new)
            elif isinstance(comp, DynamicAddressComponent):
                new = IdxChm.build(comp, new)

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
        ```python exec="yes" source="material-block" session="core"
        from genjax import ChoiceMapBuilder as C

        chm = C["x"].set(3.0)
        print(chm.render_html())
        ```

        (**Getting submaps**) Hierarchical choice maps support `__call__`, which allows for the retrieval of _submaps_ at addresses:
        ```python exec="yes" source="material-block" session="core"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0)
        submap = chm("x")
        print(submap.render_html())
        ```

        (**Getting values**) Choice maps support `__getitem__`, which allows for the retrieval of _values_ at addresses:
        ```python exec="yes" source="material-block" session="core"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0)
        value = chm["x", "y"]
        print(value)
        ```

        (**Making vectorized choice maps**) Choice maps can be constructed using `jax.vmap`:
        ```python exec="yes" source="material-block" session="core"
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
        return _empty

    @classmethod
    def value(cls, v: T) -> "ValueChm[T]":
        return ValueChm(v)

    @classmethod
    def idx(cls, addr: AddressComponent, v: Any) -> "ChoiceMap":
        chm = v if isinstance(v, ChoiceMap) else ChoiceMap.value(v)
        if isinstance(addr, StaticAddressComponent):
            return StaticChm.build(addr, chm)
        else:
            return IdxChm.build(addr, chm)

    @classmethod
    def d(cls, d: dict[Any, Any]) -> "ChoiceMap":
        start = ChoiceMap.empty()
        if d:
            for k, v in d.items():
                start = ChoiceMapBuilder.a(k, v) ^ start
        return start

    @classmethod
    def kw(cls, **kwargs) -> "ChoiceMap":
        return ChoiceMap.d(kwargs)

    ######################
    # Combinator methods #
    ######################

    def filter(self, selection: Selection) -> "ChoiceMap":
        """Filter the choice map on the `Selection`. The resulting choice map only contains the addresses in the selection.

        Examples:
            ```python exec="yes" source="material-block" session="core"
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
        return FilteredChm.build(selection, self)

    def indexed(self, addr: AddressComponent) -> "ChoiceMap":
        return ChoiceMap.idx(addr, self)

    def mask(self, f: Flag) -> "ChoiceMap":
        return MaskChm.build(f, self)

    def merge(self, other: "ChoiceMap") -> "ChoiceMap":
        return self ^ other

    def get_selection(self) -> Selection:
        """Convert a `ChoiceMap` to a `Selection`."""
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
        """Access the `ChoiceMap.AddressIndex` mutation interface. This allows
        users to take an existing choice map, and mutate it _functionally_.

        Examples:
        ```python exec="yes" source="material-block" session="core"
        chm = C["x", "y"].set(3.0)
        chm = chm.at["x", "y"].set(4.0)
        print(chm["x", "y"])
        ```
        """
        return AddressIndex(self, [])


@Pytree.dataclass
class EmptyChm(ChoiceMap):
    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        return self

    def static_is_empty(self) -> Bool:
        return True


_empty = EmptyChm()


@Pytree.dataclass
class ValueChm(Generic[T], ChoiceMap):
    v: T

    def get_value(self) -> T:
        return self.v

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        return _empty


@Pytree.dataclass
class IdxChm(ChoiceMap):
    addr: DynamicAddressComponent
    c: ChoiceMap

    @classmethod
    def build(cls, addr: DynamicAddressComponent, chm: ChoiceMap) -> ChoiceMap:
        return _empty if chm.static_is_empty() else IdxChm(addr, chm)

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
                    Flag(check[addr]), jtu.tree_map(lambda v: v[addr], self.c)
                )
                if jnp.array(check, copy=False).shape
                else self.c.mask(Flag(check))
            )


@Pytree.dataclass
class StaticChm(ChoiceMap):
    addr: StaticAddressComponent = Pytree.static()
    c: ChoiceMap = Pytree.field()

    @classmethod
    def build(
        cls,
        addr: StaticAddressComponent,
        c: ChoiceMap,
    ) -> ChoiceMap:
        return _empty if c.static_is_empty() else StaticChm(addr, c)

    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        check = Flag(addr == self.addr)
        return self.c.mask(check)


@Pytree.dataclass
class XorChm(ChoiceMap):
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
    flag: Flag
    c: ChoiceMap

    @classmethod
    def build(
        cls,
        flag: Flag,
        c: ChoiceMap,
    ) -> ChoiceMap:
        return (
            c
            if c.static_is_empty()
            else c
            if flag.concrete_true()
            else _empty
            if flag.concrete_false()
            else MaskChm(flag, c)
        )

    def get_value(self) -> Any:
        v = self.c.get_value()
        return Mask.maybe_none(self.flag, v)

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        return submap.mask(self.flag)


@Pytree.dataclass
class FilteredChm(ChoiceMap):
    selection: Selection
    c: ChoiceMap

    @classmethod
    def build(cls, selection: Selection, chm: ChoiceMap) -> ChoiceMap:
        return _empty if chm.static_is_empty() else FilteredChm(selection, chm)

    def get_value(self) -> Any:
        v = self.c.get_value()
        sel_check = self.selection[()]
        return Mask.maybe_none(sel_check, v)

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        subselection = self.selection(addr)
        return submap.filter(subselection)
