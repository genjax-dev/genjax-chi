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
    String,
    typecheck,
)

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
            if isinstance(comp, ExtendedStaticAddressComponent):
                sel = Selection.str(comp, sel)
            elif isinstance(comp, DynamicAddressComponent):
                sel = Selection.idx(comp, sel)
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

    def __or__(self, other: "Selection") -> "Selection":
        return select_or(self, other)

    def __and__(self, other):
        return select_and(self, other)

    def __invert__(self) -> "Selection":
        return select_complement(self)

    @typecheck
    def __call__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ):
        addr = addr if isinstance(addr, tuple) else (addr,)
        subselection = self
        for comp in addr:
            subselection = subselection.get_subselection(comp)
        return subselection

    @typecheck
    def __getitem__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ) -> Flag:
        subselection = self(addr)
        return subselection.check()

    @typecheck
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

    #################################################
    # Convenient syntax for constructing selections #
    #################################################

    @classmethod
    @typecheck
    def all(_cls) -> "Selection":
        return select_all()

    @classmethod
    @typecheck
    def str(
        _cls, comp: ExtendedStaticAddressComponent, sel: "Selection"
    ) -> "Selection":
        return select_static(comp, sel)

    @classmethod
    @typecheck
    def idx(_cls, comp: DynamicAddressComponent, sel: "Selection") -> "Selection":
        return select_idx(comp, sel)

    @classmethod
    @typecheck
    def maybe(_cls, flag: Flag, s: "Selection") -> "Selection":
        return select_defer(flag, s)


#######################
# Selection functions #
#######################


@Pytree.dataclass
class AllSel(Selection):
    def check(self) -> Flag:
        return Flag(True)

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        return AllSel()


def select_all():
    return AllSel()


@Pytree.dataclass
class DeferSel(Selection):
    flag: Flag
    s: Selection

    def check(self) -> Flag:
        ch = self.s.check()
        return self.flag.and_(ch)

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining = self.s(addr)
        return select_defer(self.flag, remaining)


@typecheck
def select_defer(
    flag: Flag,
    s: Selection,
) -> Selection:
    return DeferSel(flag, s)


@Pytree.dataclass
class CompSel(Selection):
    s: Selection

    def check(self) -> Flag:
        return self.s.check().not_()

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining = self.s(addr)
        return select_complement(remaining)


@typecheck
def select_complement(
    s: Selection,
) -> Selection:
    return CompSel(s)


def select_none():
    return select_complement(select_all())


@Pytree.dataclass
class StaticSel(Selection):
    addr: ExtendedStaticAddressComponent = Pytree.static()
    s: Selection = Pytree.field()

    def check(self) -> Flag:
        return Flag(False)

    def get_subselection(self, addr: EllipsisType | AddressComponent) -> Selection:
        check = Flag(addr == self.addr or isinstance(addr, EllipsisType))
        return select_defer(check, self.s)


@typecheck
def select_static(
    addr: EllipsisType | StaticAddressComponent,
    s: Selection,
) -> Selection:
    return StaticSel(addr, s)


@Pytree.dataclass
class IdxSel(Selection):
    idxs: DynamicAddressComponent
    s: Selection

    def check(self) -> Flag:
        return Flag(False)

    def get_subselection(self, addr: EllipsisType | AddressComponent) -> Selection:
        if isinstance(addr, EllipsisType):
            return self.s

        if not isinstance(addr, DynamicAddressComponent):
            return select_none()

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
            return select_defer(check, self.s)


@typecheck
def select_idx(
    sidx: DynamicAddressComponent,
    s: Selection,
) -> Selection:
    return IdxSel(sidx, s)


@Pytree.dataclass
class AndSel(Selection):
    s1: Selection
    s2: Selection

    def check(self) -> Flag:
        return self.s1.check().and_(self.s2.check())

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return select_and(remaining1, remaining2)


@typecheck
def select_and(
    s1: Selection,
    s2: Selection,
) -> Selection:
    return AndSel(s1, s2)


@Pytree.dataclass
class OrSel(Selection):
    s1: Selection
    s2: Selection

    def check(self) -> Flag:
        return self.s1.check().or_(self.s2.check())

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return select_or(remaining1, remaining2)


@typecheck
def select_or(
    s1: Selection,
    s2: Selection,
) -> Selection:
    return OrSel(s1, s2)


@Pytree.dataclass
class ChmSel(Selection):
    c: "ChoiceMap"

    def check(self) -> Flag:
        return check_none(self.c.get_value())

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        submap = self.c.get_submap(addr)
        return select_choice_map(submap)


@typecheck
def select_choice_map(
    c: "ChoiceMap",
) -> Selection:
    return ChmSel(c)


###############
# Choice maps #
###############


@dataclass
class ChoiceMapNoValueAtAddress(Exception):
    subaddr: Any


@Pytree.dataclass
class _ChoiceMapBuilder(Pytree):
    addr: ExtendedAddress = ()

    @typecheck
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
            return choice_map_empty

    def n(self) -> "ChoiceMap":
        return choice_map_empty

    def v(self, v) -> "ChoiceMap":
        return ChoiceMap.value(v)

    def d(self, d: dict[Any, Any]) -> "ChoiceMap":
        return ChoiceMap.d(d)

    def kw(self, **kwargs) -> "ChoiceMap":
        return ChoiceMap.kw(**kwargs)

    @typecheck
    def a(
        self, addr: ExtendedAddressComponent | ExtendedAddress, v: Any
    ) -> "ChoiceMap":
        addr = addr if isinstance(addr, tuple) else (addr,)
        new = ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        for comp in reversed(addr):
            if isinstance(comp, StaticAddressComponent):
                new = ChoiceMap.str(comp, new)
            elif isinstance(comp, DynamicAddressComponent):
                new = ChoiceMap.idx(comp, new)
        return new


ChoiceMapBuilder = _ChoiceMapBuilder()


def check_none(v) -> Flag:
    if v is None:
        return Flag(False)
    elif isinstance(v, Mask):
        return v.flag
    else:
        return Flag(True)


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

    @typecheck
    def has_value(self) -> Flag:
        return check_none(self.get_value())

    @typecheck
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
        return choice_map_filtered(selection, self)

    def merge(self, other):
        return choice_map_xor(self, other)

    def get_selection(self) -> Selection:
        """Convert a `ChoiceMap` to a `Selection`."""
        return select_choice_map(self)

    @typecheck
    def static_is_empty(self) -> Bool:
        return False

    ###########
    # Dunders #
    ###########

    def __xor__(self, other):
        return self.merge(other)

    def __add__(self, other):
        return choice_map_or(self, other)

    @typecheck
    def __call__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ):
        addr = addr if isinstance(addr, tuple) else (addr,)
        submap = self
        for comp in addr:
            submap = submap.get_submap(comp)
        return submap

    @typecheck
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

    @typecheck
    def __contains__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ):
        addr = addr if isinstance(addr, tuple) else (addr,)
        submap = self
        for comp in addr:
            submap = self.get_submap(comp)
        return submap.has_value()

    ######################################
    # Convenient syntax for construction #
    ######################################

    @classmethod
    def empty(_cls) -> "ChoiceMap":
        return choice_map_empty

    @classmethod
    def value(_cls, v) -> "ChoiceMap":
        return choice_map_value(v)

    @classmethod
    def maybe(_cls, f: Flag, c: "ChoiceMap") -> "ChoiceMap":
        return choice_map_masked(f, c)

    @classmethod
    def str(_cls, addr: StaticAddressComponent, v: Any) -> "ChoiceMap":
        return choice_map_static(
            addr, ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        )

    @classmethod
    def idx(_cls, addr: DynamicAddressComponent, v: Any) -> "ChoiceMap":
        return choice_map_idx(
            addr, ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        )

    @classmethod
    def d(_cls, d: dict[Any, Any]) -> "ChoiceMap":
        start = ChoiceMap.empty()
        if d:
            for k, v in d.items():
                start = ChoiceMapBuilder.a(k, v) ^ start
        return start

    @classmethod
    def kw(_cls, **kwargs) -> "ChoiceMap":
        return ChoiceMap.d(kwargs)

    ##########################
    # AddressIndex interface #
    ##########################

    @Pytree.dataclass
    class AddressIndex(Pytree):
        choice_map: "ChoiceMap"
        addrs: list[Address]

        def __getitem__(
            self, addr: AddressComponent | Address
        ) -> "ChoiceMap.AddressIndex":
            addr = addr if isinstance(addr, tuple) else (addr,)
            return ChoiceMap.AddressIndex(
                self.choice_map,
                [*self.addrs, addr],
            )

        def set(self, v):
            new = self.choice_map
            for addr in self.addrs:
                new = ChoiceMapBuilder.a(addr, v) + new
            return new

        @property
        def at(self) -> "ChoiceMap.AddressIndex":
            return self

        def filter(self):
            sels = map(lambda addr: SelectionBuilder[addr], self.addrs)
            or_sel = reduce(or_, sels)
            return self.choice_map.filter(or_sel)

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
        return ChoiceMap.AddressIndex(self, [])


@Pytree.dataclass
class EmptyChm(ChoiceMap):
    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        return EmptyChm()

    def static_is_empty(self) -> Bool:
        return True


choice_map_empty = EmptyChm()


@Pytree.dataclass
class ValueChm(ChoiceMap):
    v: Any

    def get_value(self) -> Any:
        return self.v

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        return choice_map_empty


@typecheck
def choice_map_value(
    v: Any,
) -> ChoiceMap:
    return ValueChm(v)


@Pytree.dataclass
class IdxChm(ChoiceMap):
    addr: DynamicAddressComponent
    c: ChoiceMap

    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        if addr is Ellipsis:
            return self.c

        elif not isinstance(addr, DynamicAddressComponent):
            return choice_map_empty

        else:

            def check_fn(idx, addr) -> BoolArray:
                return jnp.array(idx == addr, copy=False)

            check = (
                jax.vmap(check_fn, in_axes=(None, 0))(addr, self.addr)
                if jnp.array(self.addr, copy=False).shape
                else check_fn(addr, self.addr)
            )

            return (
                choice_map_masked(
                    Flag(check[addr]), jtu.tree_map(lambda v: v[addr], self.c)
                )
                if jnp.array(check, copy=False).shape
                else choice_map_masked(Flag(check), self.c)
            )


@typecheck
def choice_map_idx(
    addr: DynamicAddressComponent,
    c: ChoiceMap,
) -> ChoiceMap:
    return choice_map_empty if c.static_is_empty() else IdxChm(addr, c)


@Pytree.dataclass
class StaticChm(ChoiceMap):
    addr: AddressComponent = Pytree.static()
    c: ChoiceMap = Pytree.field()

    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        check = Flag(addr == self.addr)
        return choice_map_masked(check, self.c)


@typecheck
def choice_map_static(
    addr: AddressComponent,
    c: ChoiceMap,
) -> ChoiceMap:
    return choice_map_empty if c.static_is_empty() else StaticChm(addr, c)


@Pytree.dataclass
class XorChm(ChoiceMap):
    c1: ChoiceMap
    c2: ChoiceMap

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
        return choice_map_xor(remaining_1, remaining_2)


@typecheck
def choice_map_xor(
    c1: ChoiceMap,
    c2: ChoiceMap,
) -> ChoiceMap:
    match (c1.static_is_empty(), c2.static_is_empty()):
        case True, True:
            return choice_map_empty
        case _, True:
            return c1
        case True, _:
            return c2
        case _:
            return XorChm(c1, c2)


@Pytree.dataclass
class OrChm(ChoiceMap):
    c1: ChoiceMap
    c2: ChoiceMap

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

        return choice_map_or(submap1, submap2)


@typecheck
def choice_map_or(
    c1: ChoiceMap,
    c2: ChoiceMap,
) -> ChoiceMap:
    match (c1.static_is_empty(), c2.static_is_empty()):
        case True, True:
            return choice_map_empty
        case _, True:
            return c1
        case True, _:
            return c2
        case _:
            return OrChm(c1, c2)


@Pytree.dataclass
class MaskChm(ChoiceMap):
    flag: Flag
    c: ChoiceMap

    def get_value(self) -> Any:
        v = self.c.get_value()
        return Mask.maybe_none(self.flag, v)

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        return choice_map_masked(self.flag, submap)


@typecheck
def choice_map_masked(
    flag: Flag,
    c: ChoiceMap,
) -> ChoiceMap:
    return (
        c
        if c.static_is_empty()
        else c
        if flag.concrete_true()
        else choice_map_empty
        if flag.concrete_false()
        else MaskChm(flag, c)
    )


@Pytree.dataclass
class FilteredChm(ChoiceMap):
    selection: Selection
    c: ChoiceMap

    def get_value(self) -> Any:
        v = self.c.get_value()
        sel_check = self.selection[()]
        return Mask.maybe_none(sel_check, v)

    def get_submap(self, addr: ExtendedAddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        subselection = self.selection(addr)
        return choice_map_filtered(subselection, submap)


@typecheck
def choice_map_filtered(
    selection: Selection,
    c: ChoiceMap,
) -> ChoiceMap:
    return choice_map_empty if c.static_is_empty() else FilteredChm(selection, c)
