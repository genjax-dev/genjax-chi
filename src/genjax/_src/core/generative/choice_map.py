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
    staged_and,
    staged_err,
    staged_not,
    staged_or,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Bool,
    BoolArray,
    EllipsisType,
    Int,
    IntArray,
    List,
    Optional,
    String,
    Tuple,
    Union,
    static_check_bool,
    typecheck,
)

register_exclusion(__file__)

#################
# Address types #
#################

StaticAddressComponent = String
DynamicAddressComponent = Int | IntArray
AddressComponent = StaticAddressComponent | DynamicAddressComponent
Address = Tuple[()] | Tuple[AddressComponent, ...]
StaticAddress = Tuple[()] | Tuple[StaticAddressComponent, ...]
ExtendedStaticAddressComponent = StaticAddressComponent | EllipsisType
ExtendedAddressComponent = ExtendedStaticAddressComponent | DynamicAddressComponent
ExtendedStaticAddress = Tuple[()] | Tuple[ExtendedStaticAddressComponent, ...]
ExtendedAddress = Tuple[()] | Tuple[ExtendedAddressComponent, ...]


##############
# Selections #
##############


###############################
# Selection builder interface #
###############################


@Pytree.dataclass
class _SelectionBuilder(Pytree):
    def __getitem__(self, addr_comps):
        if not isinstance(addr_comps, Tuple):
            addr_comps = (addr_comps,)

        sel = Selection.all()
        for comp in reversed(addr_comps):
            if isinstance(comp, StaticAddressComponent | EllipsisType):
                sel = Selection.str(comp, sel)
            elif isinstance(comp, DynamicAddressComponent):
                sel = Selection.idx(comp, sel)
        return sel


SelectionBuilder = _SelectionBuilder()


@Pytree.dataclass
class Selection(ProjectProblem):
    selection_function: "SelectionFunction"

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
            subselection = subselection.selection_function.get_subselection(comp)
        return subselection

    @typecheck
    def __getitem__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ) -> Bool | BoolArray:
        subselection = self(addr)
        return subselection.selection_function.check()

    @typecheck
    def __contains__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ) -> Bool | BoolArray:
        return self[addr]

    def check(self) -> Bool | BoolArray:
        return self.selection_function.check()

    #################################################
    # Convenient syntax for constructing selections #
    #################################################

    @classmethod
    @typecheck
    def all(cls) -> "Selection":
        return select_all()

    @classmethod
    @typecheck
    def str(cls, comp: ExtendedStaticAddressComponent, sel: "Selection") -> "Selection":
        return select_static(comp, sel)

    @classmethod
    @typecheck
    def idx(cls, comp: DynamicAddressComponent, sel: "Selection") -> "Selection":
        return select_idx(comp, sel)

    @classmethod
    @typecheck
    def maybe(cls, flag: Union[Bool, BoolArray], s: "Selection") -> "Selection":
        return select_defer(flag, s)


#######################
# Selection functions #
#######################


class SelectionFunction(Pytree):
    """
    The signature here is inspired by monadic parser combinators.
    For instance: https://www.cmi.ac.in/~spsuresh/teaching/prgh15/papers/monadic-parsing.pdf
    SelectionFunction = Callable[
        [AddressComponent],
        Tuple[BoolArray, Selection],
    ]
    """

    @abstractmethod
    def check(self) -> Bool | BoolArray:
        pass

    @abstractmethod
    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        pass


@Pytree.dataclass
class AllSelFn(SelectionFunction):
    def check(self) -> Bool | BoolArray:
        return True

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        return Selection(AllSelFn())


def select_all():
    return Selection(AllSelFn())


@Pytree.dataclass
class DeferSelFn(SelectionFunction):
    flag: Union[Bool, BoolArray]
    s: Selection

    def check(self) -> Bool | BoolArray:
        ch = self.s.check()
        return staged_and(self.flag, ch)

    def get_subselection(self, addr: ExtendedAddressComponent) -> Selection:
        remaining = self.s(addr)
        return select_defer(self.flag, remaining)


@typecheck
def select_defer(
    flag: Union[Bool, BoolArray],
    s: Selection,
) -> Selection:
    return Selection(DeferSelFn(flag, s))


@Pytree.dataclass
class CompSelFn(SelectionFunction):
    s: Selection

    def check(self) -> Bool | BoolArray:
        ch = self.s.check()
        return staged_not(ch)

    def get_subselection(self, addr: AddressComponent) -> Selection:
        remaining = self.s(addr)
        return select_complement(remaining)


@typecheck
def select_complement(
    s: Selection,
) -> Selection:
    return Selection(CompSelFn(s))


def select_none():
    return select_complement(select_all())


@Pytree.dataclass
class StaticSelFn(SelectionFunction):
    addr: StaticAddressComponent = Pytree.static()
    s: Selection = Pytree.field()

    def check(self) -> Bool | BoolArray:
        return False

    def get_subselection(self, addr: EllipsisType | AddressComponent) -> Selection:
        check = addr == self.addr or isinstance(addr, EllipsisType)
        return select_defer(check, self.s)


@typecheck
def select_static(
    addr: EllipsisType | StaticAddressComponent,
    s: Selection,
) -> Selection:
    return Selection(StaticSelFn(addr, s))


@Pytree.dataclass
class IdxSelFn(SelectionFunction):
    idx: DynamicAddressComponent
    s: Selection

    def check(self) -> Bool | BoolArray:
        return False

    def get_subselection(self, addr: EllipsisType | AddressComponent) -> Selection:
        if isinstance(addr, EllipsisType):
            return self.s

        if not isinstance(addr, DynamicAddressComponent):
            return select_none()

        else:

            def check_fn(v):
                return staged_and(
                    v,
                    jnp.any(v == self.idx),
                )

            check = (
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
    return Selection(IdxSelFn(sidx, s))


@Pytree.dataclass
class AndSelFn(SelectionFunction):
    s1: Selection
    s2: Selection

    def check(self) -> Bool | BoolArray:
        check1 = self.s1.check()
        check2 = self.s2.check()
        return staged_and(check1, check2)

    def get_subselection(self, addr: AddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return select_and(remaining1, remaining2)


@typecheck
def select_and(
    s1: Selection,
    s2: Selection,
) -> Selection:
    return Selection(AndSelFn(s1, s2))


@Pytree.dataclass
class OrSelFn(SelectionFunction):
    s1: Selection
    s2: Selection

    def check(self) -> Bool | BoolArray:
        check1 = self.s1.check()
        check2 = self.s2.check()
        return staged_or(check1, check2)

    def get_subselection(self, addr: AddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return select_or(remaining1, remaining2)


@typecheck
def select_or(
    s1: Selection,
    s2: Selection,
) -> Selection:
    return Selection(OrSelFn(s1, s2))


@Pytree.dataclass
class ChmSelFn(SelectionFunction):
    c: "ChoiceMap"

    def check(self) -> Bool | BoolArray:
        return check_none(self.c.get_value())

    def get_subselection(self, addr: AddressComponent) -> Selection:
        submap = self.c.get_submap(addr)
        return select_choice_map(submap)


@typecheck
def select_choice_map(
    c: "ChoiceMap",
) -> Selection:
    return Selection(ChmSelFn(c))


###############
# Choice maps #
###############


@dataclass
class ChoiceMapNoValueAtAddress(Exception):
    subaddr: Any


@Pytree.dataclass
class _ChoiceMapBuilder(Pytree):
    addr: Optional[Address]

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

    def d(self, d: dict) -> "ChoiceMap":
        return ChoiceMap.d(d)

    def kw(self, **kwargs) -> "ChoiceMap":
        return ChoiceMap.kw(**kwargs)

    @typecheck
    def a(self, addr: ExtendedAddress, v: Any) -> "ChoiceMap":
        new = ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        for comp in reversed(addr):
            if isinstance(comp, ExtendedStaticAddressComponent):
                new = ChoiceMap.str(comp, new)
            else:
                new = ChoiceMap.idx(comp, new)
        return new


ChoiceMapBuilder = _ChoiceMapBuilder(None)


def check_none(v):
    if v is None:
        return False
    elif isinstance(v, Mask):
        return v.flag
    else:
        return True


@Pytree.dataclass
class ChoiceMap(Sample, Constraint):
    """
    The type `ChoiceMap` denotes a map-like value which can be sampled from a generative function.

    Generative functions which utilize map-like representations often support a notion of _addressing_,
    allowing the invocation of generative function callees, whose choices become addressed random choices
    in the caller's choice map.
    """

    choice_map_fn: "ChoiceMapFunction"

    #######################
    # Map-like interfaces #
    #######################

    def get_value(self) -> Any:
        return self.choice_map_fn.get_value()

    @typecheck
    def has_value(self) -> Bool | BoolArray:
        return check_none(self.choice_map_fn.get_value())

    @typecheck
    def get_submap(
        self,
        addr: ExtendedAddressComponent,
    ) -> "ChoiceMap":
        return self.choice_map_fn.get_submap(addr)

    @typecheck
    def filter(self, selection: Selection) -> "ChoiceMap":
        """Filter the choice map on the `Selection`. The resulting choice map only contains the addresses in the selection.

        Examples:
            ```python exec="yes" source="material-block" session="core"
            import jax
            import genjax
            from genjax import bernoulli, Selection

            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.get_sample()
            selection = Selection.at["x"]
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
            submap = submap.choice_map_fn.get_submap(comp)
        return submap

    @typecheck
    def __getitem__(
        self,
        addr: ExtendedAddressComponent | ExtendedAddress,
    ):
        addr = addr if isinstance(addr, tuple) else (addr,)
        submap = self(addr)
        v = submap.get_value()
        # Aggressively unwrap functional types.
        # This _can_ throw an error, but the user
        # is asking for it.
        # TODO: providing error handling here
        # to make what is happening transparent.
        if isinstance(v, Mask):
            v = v.unmask()
        if isinstance(v, Sum):
            v = v.maybe_collapse()
            v = v.unmask() if isinstance(v, Mask) else v
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
    def empty(cls) -> "ChoiceMap":
        return choice_map_empty

    @classmethod
    def value(cls, v) -> "ChoiceMap":
        return choice_map_value(v)

    @classmethod
    def maybe(cls, f: BoolArray, c: "ChoiceMap") -> "ChoiceMap":
        return choice_map_masked(f, c)

    @classmethod
    def str(cls, addr: StaticAddressComponent, v: Any) -> "ChoiceMap":
        return choice_map_static(
            addr, ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        )

    @classmethod
    def idx(cls, addr: DynamicAddressComponent, v: Any) -> "ChoiceMap":
        return choice_map_idx(
            addr, ChoiceMap.value(v) if not isinstance(v, ChoiceMap) else v
        )

    @classmethod
    def d(cls, d: dict) -> "ChoiceMap":
        start = ChoiceMap.empty()
        if d:
            for k, v in d.items():
                start = ChoiceMapBuilder.a(k, v) ^ start
        return start

    @classmethod
    def kw(cls, **kwargs) -> "ChoiceMap":
        return ChoiceMap.d(kwargs)

    # NOTE: this only allows dictionaries with static keys
    # a.k.a. strings -- not jax.arrays -- for now.
    def addr_fn(self, addr_fn: dict):
        return choice_map_address_function(addr_fn, self)

    ##########################
    # AddressIndex interface #
    ##########################

    @Pytree.dataclass
    class AddressIndex(Pytree):
        choice_map: "ChoiceMap"
        addrs: List[Address]

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
        return ChoiceMap.AddressIndex(self, [])


class ChoiceMapFunction(Pytree):
    """
    ChoiceMapFunction = Callable[
        [AddressComponent],
        Tuple[Optional[Any], ChoiceMap],
        # The elements of this type are, given an address component:
        # 1. Value at this node, if there is one
        # 2. The sub choice map at this node.
    ]
    """

    @abstractmethod
    def get_value(self) -> Optional[Any]:
        pass

    @abstractmethod
    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        pass

    def __call__(self, addr: AddressComponent):
        value = self.get_value()
        submap = self.get_submap(addr)
        return value, submap


@Pytree.dataclass
class EmptyChmFn(ChoiceMapFunction):
    def get_value(self) -> Any:
        return None

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        return ChoiceMap(EmptyChmFn())


choice_map_empty = ChoiceMap(EmptyChmFn())


@Pytree.dataclass
class ValueChmFn(ChoiceMapFunction):
    v: Any

    def get_value(self) -> Optional[Any]:
        return self.v

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        return choice_map_empty


@typecheck
def choice_map_value(
    v: Any,
) -> ChoiceMap:
    return ChoiceMap(ValueChmFn(v))


@Pytree.dataclass
class IdxChmFn(ChoiceMapFunction):
    addr: DynamicAddressComponent
    c: ChoiceMap

    def get_value(self) -> Optional[Any]:
        return None

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
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
                choice_map_masked(check[addr], jtu.tree_map(lambda v: v[addr], self.c))
                if jnp.array(check, copy=False).shape
                else choice_map_masked(check, self.c)
            )


@typecheck
def choice_map_idx(
    addr: DynamicAddressComponent,
    c: ChoiceMap,
) -> ChoiceMap:
    return ChoiceMap(IdxChmFn(addr, c))


@Pytree.dataclass
class StaticChmFn(ChoiceMapFunction):
    addr: AddressComponent = Pytree.static()
    c: ChoiceMap = Pytree.field()

    def get_value(self) -> Optional[Any]:
        return None

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        check = addr == self.addr
        return choice_map_masked(check, self.c)


@typecheck
def choice_map_static(
    addr: AddressComponent,
    c: ChoiceMap,
) -> ChoiceMap:
    return ChoiceMap(StaticChmFn(addr, c))


@Pytree.dataclass
class XorChmFn(ChoiceMapFunction):
    c1: ChoiceMap
    c2: ChoiceMap

    def get_value(self) -> Optional[Any]:
        check1 = self.c1.has_value()
        check2 = self.c2.has_value()
        err_check = staged_and(check1, check2)
        staged_err(
            err_check,
            f"The disjoint union of two choice maps have a value collision:\nc1 = {self.c1}\nc2 = {self.c2}",
        )
        v1 = self.c1.get_value()
        v2 = self.c2.get_value()

        def pair_bool_to_idx(bool1, bool2):
            return (1 * bool1 + 2 * bool2 - 3 * (bool1 & bool2)) - 1

        idx = pair_bool_to_idx(check1, check2)
        return Sum.maybe_none(idx, [v1, v2])

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        remaining_1 = self.c1.get_submap(addr)
        remaining_2 = self.c2.get_submap(addr)
        return choice_map_xor(remaining_1, remaining_2)


@typecheck
def choice_map_xor(c1: ChoiceMap, c2: ChoiceMap):
    return ChoiceMap(XorChmFn(c1, c2))


@Pytree.dataclass
class OrChmFn(ChoiceMapFunction):
    c1: ChoiceMap
    c2: ChoiceMap

    def get_value(self) -> Optional[Any]:
        check1 = self.c1.has_value()
        check2 = self.c2.has_value()
        v1 = self.c1.get_value()
        v2 = self.c2.get_value()

        def pair_bool_to_idx(first, second):
            output = -1 + first + 2 * (staged_not(first) & second)
            return output

        idx = pair_bool_to_idx(check1, check2)
        return Sum.maybe_none(idx, [v1, v2])

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        submap1 = self.c1.get_submap(addr)
        submap2 = self.c2.get_submap(addr)

        return choice_map_or(submap1, submap2)


@typecheck
def choice_map_or(
    c1: ChoiceMap,
    c2: ChoiceMap,
) -> ChoiceMap:
    return ChoiceMap(OrChmFn(c1, c2))


@Pytree.dataclass
class MaskChmFn(ChoiceMapFunction):
    flag: Bool | BoolArray
    c: ChoiceMap

    def get_value(self) -> Optional[Any]:
        v = self.c.get_value()
        return Mask.maybe_none(self.flag, v)

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        return choice_map_masked(self.flag, submap)


@typecheck
def choice_map_masked(
    flag: Bool | BoolArray,
    c: ChoiceMap,
) -> ChoiceMap:
    if static_check_bool(flag):
        if flag:
            return c
        else:
            return choice_map_empty
    else:
        return ChoiceMap(MaskChmFn(flag, c))


@Pytree.dataclass
class FilteredChmFn(ChoiceMapFunction):
    selection: Selection
    c: ChoiceMap

    def get_value(self) -> Optional[Any]:
        v = self.c.get_value()
        sel_check = () in self.selection
        return Mask.maybe(sel_check, v)

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        submap = self.c.get_submap(addr)
        subselection = self.selection(addr)
        return choice_map_filtered(subselection, submap)


@typecheck
def choice_map_filtered(
    selection: Selection,
    c: ChoiceMap,
) -> ChoiceMap:
    return ChoiceMap(FilteredChmFn(selection, c))


@Pytree.dataclass
class AddrFnChmFn(ChoiceMapFunction):
    c: ChoiceMap
    addr_fn: dict = Pytree.static()

    def get_value(self) -> Bool | BoolArray:
        mapped = self.addr_fn.get((), ())
        if mapped:
            submap = self.c.get_submap(mapped)
            return submap.get_value()
        else:
            return self.c.get_value()

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        mapped = self.addr_fn.get(addr, addr)
        return self.c.get_submap(mapped)


@typecheck
def choice_map_address_function(
    addr_fn: dict,
    c: ChoiceMap,
):
    return ChoiceMap(AddrFnChmFn(c, addr_fn))
