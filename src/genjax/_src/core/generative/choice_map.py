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

from functools import reduce
from operator import or_

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap
from penzai import pz

from genjax._src.core.generative.core import Constraint, Sample, UpdateSpec
from genjax._src.core.generative.functional_types import Mask, Sum
from genjax._src.core.interpreters.staging import (
    staged_and,
    staged_err,
    staged_not,
    staged_or,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Bool,
    BoolArray,
    Callable,
    EllipsisType,
    Int,
    List,
    String,
    Tuple,
    Union,
    static_check_is_concrete,
    typecheck,
)

#################
# Address types #
#################

StaticAddressComponent = String
DynamicAddressComponent = ArrayLike
AddressComponent = Union[
    Tuple[()],
    EllipsisType,
    StaticAddressComponent,
    DynamicAddressComponent,
]
Address = Union[
    AddressComponent,
    Tuple[AddressComponent, ...],
]

##############
# Selections #
##############

# NOTE: the signature here is inspired by monadic parser combinators.
# For instance:
# https://www.cmi.ac.in/~spsuresh/teaching/prgh15/papers/monadic-parsing.pdf
SelectionFunction = Callable[
    [AddressComponent],
    Tuple[BoolArray, "Selection"],
]


# NOTE: normal class properties are deprecated in Python 3.11,
# so here's our simple custom version.
class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


@Pytree.dataclass
class Selection(Pytree):
    has_addr: SelectionFunction
    info: String = Pytree.static()

    def __or__(self, other: "Selection") -> "Selection":
        return select_or(self, other)

    def __rshift__(self, other):
        return select_and_then(self, other)

    def __and__(self, other):
        return select_and(self, other)

    def __invert__(self) -> "Selection":
        return select_complement(self)

    def complement(self) -> "Selection":
        return select_complement(self)

    @classmethod
    def with_info(cls, info: String):
        return lambda fn: Selection(fn, info)

    @classmethod
    def get_address_head(cls, addr: Address) -> Address:
        if isinstance(addr, tuple) and addr:
            return addr[0]
        else:
            return addr

    @classmethod
    def get_address_tail(cls, addr: Address) -> Address:
        if isinstance(addr, tuple):
            return addr[1:]
        else:
            return ()

    # Take a single step in the selection process, by focusing
    # the Selection on the head of the address.
    def step(self, addr: AddressComponent):
        _, remaining = self.has_addr(addr)
        return remaining

    def has(self, addr: Address) -> BoolArray:
        check, _ = self.has_addr(addr)
        return check

    def do(self, addr: Address) -> BoolArray:
        if addr:
            head = Selection.get_address_head(addr)
            tail = Selection.get_address_tail(addr)
            remaining = self.step(head)
            return remaining.do(tail)
        else:
            return self.has(())

    def __getitem__(self, addr: Address) -> BoolArray:
        return self.do(addr)

    def __contains__(self, addr: Address) -> BoolArray:
        return self.has(addr)

    def __str__(self):
        return f"Selection({self.info})"

    def __repr__(self):
        return f"Selection({self.info})"

    @classproperty
    def n(cls) -> "Selection":
        return select_none

    @classproperty
    def a(cls) -> "Selection":
        return select_all

    @classmethod
    @typecheck
    def s(cls, comp: StaticAddressComponent) -> "Selection":
        return select_static(comp)

    @classmethod
    @typecheck
    def idx(cls, comp: DynamicAddressComponent) -> "Selection":
        return select_idx(comp)

    @classmethod
    @typecheck
    def m(cls, flag: Union[Bool, BoolArray], s: "Selection") -> "Selection":
        return select_defer(flag, s)

    @classmethod
    @typecheck
    def r(cls, selections):
        return reduce(or_, selections)

    @classmethod
    @typecheck
    def f(cls, *addrs: Address):
        return reduce(or_, (cls.at[addr] for addr in addrs))

    #####################
    # Builder interface #
    #####################

    @Pytree.dataclass
    class SelectionBuilder(Pytree):
        def __getitem__(self, addr_comps):
            if not isinstance(addr_comps, Tuple):
                addr_comps = (addr_comps,)

            def _comp_to_sel(comp: AddressComponent):
                if isinstance(comp, EllipsisType):
                    return Selection.a
                elif isinstance(comp, DynamicAddressComponent):
                    return Selection.idx(comp)
                # TODO: make this work, for slices where we can make it work.
                elif isinstance(comp, slice):
                    idxs = jnp.arange(comp)
                    return vmap(Selection.idx)(idxs)
                elif isinstance(comp, StaticAddressComponent):
                    return Selection.s(comp)
                else:
                    raise Exception(
                        f"Provided address component {comp} is not a valid address component type."
                    )

            sel = _comp_to_sel(addr_comps[-1])
            for comp in reversed(addr_comps[:-1]):
                sel = select_and_then(_comp_to_sel(comp), sel)
            return sel

    @classproperty
    def at(cls) -> "Selection.SelectionBuilder":
        """Access the `SelectionBuilder` interface, which supports indexing to specify a selection.

        Examples:
            Basic usage might involve creating singular selection instances:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import Selection as S

            console = genjax.console()

            # Create a selection.
            s = S.at[0, "x"]

            # Check if an address is in a selection.
            check = s[0, "x", "y"]

            print(console.render(check))
            ```

            Instances of `Selection` support a full algebra, which allows you to describe unions, intersections, and complements of the address sets represented by `Selection`.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import Selection as S

            console = genjax.console()

            # Create a selection.
            s = S.at[0, "x"] | S.at[1, "y"] | S.at["z"]

            # Check if an address is in a selection.
            check1 = s[0, "x", "y"]
            check2 = s[1, "y"]
            check3 = s["z", "y"]

            print(console.render((check1, check2, check3)))
            ```

           Selections can be complemented, using the negation operator `~`:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import Selection as S

            console = genjax.console()

            # Create a selection.
            s = S.at[0, "x"] | S.at[1, "y"] | S.at["z"]

            # Check if an address is in the complement of the selection.
            check = (~s)[1, "x", "y"]

            print(console.render(check))
            ```

           The complement of a selection is itself a selection, which can be combined with other selections using the algebra operators:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import Selection as S

            console = genjax.console()

            # Create a selection.
            s = S.at[0, "x"] | S.at[1, "y"] | S.at["z"]

            # Check if an address is in the complement of the selection.
            check = (~s | s)[0, "x"]

            print(console.render(check))
            ```

           These objects are JAX compatible, meaning you can create selections over multiple indices using `jax.vmap`, and pass these objects over `jax.jit` boundaries:
            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import genjax
            from genjax import Selection as S

            console = genjax.console()

            # Create a selection.
            s = jax.vmap(lambda idx: S.at[idx, "x"])(jnp.arange(5))

            # Check if an address is in the complement of the selection.
            check1 = (~s | s)[0, "x"]
            check2 = (~s | s)[4, "x"]

            print(console.render((check1, check2)))
            ```
        """
        return Selection.SelectionBuilder()


###############
# Choice maps #
###############


@Pytree.dataclass
class ChoiceMap(Sample, Constraint):
    """
    The type `ChoiceMap` denotes a map-like value which can be sampled from a generative function.

    Generative functions which utilize map-like representations often support a notion of _addressing_,
    allowing the invocation of generative function callees, whose choices become addressed random choices
    in the caller's choice map.
    """

    choice_map_fn: "ChoiceMapFunction"
    info: String = Pytree.static()

    def get_constraint(self) -> Constraint:
        return self

    #######################
    # Map-like interfaces #
    #######################

    def get_submap(self, addr: AddressComponent) -> "ChoiceMap":
        if isinstance(addr, tuple) and addr == ():
            return self.get_value()
        else:
            _, check, submap = self.choice_map_fn(addr)
            submap = (
                jtu.tree_map(
                    lambda v: v[addr] if jnp.array(check, copy=False).shape else v,
                    submap,
                )
                if isinstance(addr, DynamicAddressComponent)
                else submap
            )
            return submap

    def get_value(self) -> Any:
        return self()

    def has_submap(self, addr: AddressComponent) -> BoolArray:
        _, check, _ = self.choice_map_fn(addr)
        check = (
            check[addr]
            if jnp.array(check, copy=False).shape
            and isinstance(addr, DynamicAddressComponent)
            else check
        )
        return check

    def has_value(self) -> Union[Bool, BoolArray]:
        return self.has_submap(())

    def filter(self, selection: Selection) -> "ChoiceMap":
        """Filter the addresses in a choice map, returning a new choice.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli

            console = genjax.console()


            @genjax.static_gen_fn
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x


            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            choice = tr.strip()
            selection = genjax.select("x")
            filtered = choice.filter(selection)
            print(console.render(filtered))
            ```
        """
        return choice_map_filtered(selection, self)

    def merge(self, other):
        return choice_map_xor(self, other)

    # TODO: McCoy needs to test this out.
    def get_selection(self) -> Selection:
        """Convert a `ChoiceMap` to a `Selection`."""
        return select_choice_map(self)

    ##########################
    # AddressIndex interface #
    ##########################

    @Pytree.dataclass
    class AddressIndex(Pytree):
        choice_map: "ChoiceMap"
        addrs: List[Address]

        def __getitem__(self, addr: Address) -> "ChoiceMap.AddressIndex":
            return ChoiceMap.AddressIndex(
                self.choice_map,
                [*self.addrs, addr],
            )

        def set(self, v):
            new = self.choice_map
            for addr in self.addrs:
                new = ChoiceMap.a(addr, v) + new
            return new

        @property
        def at(self) -> "ChoiceMap.AddressIndex":
            return self

        def filter(self):
            sels = map(lambda addr: Selection.at[addr], self.addrs)
            or_sel = reduce(or_, sels)
            return self.choice_map.filter(or_sel)

    @property
    def at(self) -> AddressIndex:
        return ChoiceMap.AddressIndex(self, [])

    def call(self, addr: AddressComponent):
        return self.choice_map_fn(addr)

    def call_recurse(self, addr: Address):
        addr = Pytree.tree_unwrap_const(addr)
        head = Selection.get_address_head(addr)
        tail = Selection.get_address_tail(addr)
        v, check, submap = self.call(head)
        if tail:
            tail_v, tail_check, tail_submap = submap.call_recurse(tail)
            return tail_v, staged_and(check, tail_check), tail_submap
        else:
            v, check, submap = submap.call(())
            return v, check, submap

    ###########
    # Dunders #
    ###########

    def __xor__(self, other):
        return self.merge(other)

    def __add__(self, other):
        return choice_map_or(self, other)

    def __getitem__(self, addr: Address):
        addr = Pytree.tree_unwrap_const(addr)
        head = Selection.get_address_head(addr)
        tail = Selection.get_address_tail(addr)
        submap = self.get_submap(head)
        if tail:
            return submap[tail]
        else:
            return submap()

    def __contains__(self, addr: Address):
        addr = Pytree.tree_unwrap_const(addr)
        head = Selection.get_address_head(addr)
        tail = Selection.get_address_tail(addr)
        check = self.has_submap(head)
        if not tail:
            return check
        else:
            submap = self.get_submap(head)
            return submap.has_submap(tail)

    def __call__(self):
        v, _, _ = self.choice_map_fn(())
        return v

    def __str__(self):
        return f"ChoiceMap({self.info})"

    def __repr__(self):
        return f"ChoiceMap({self.info})"

    ################
    # Construction #
    ################

    @classproperty
    def n(cls) -> "ChoiceMap":
        return choice_map_empty

    @classmethod
    def v(cls, v) -> "ChoiceMap":
        return choice_map_value(v)

    @classmethod
    def s(cls, sel: Selection, v: Any) -> "ChoiceMap":
        return choice_map_selection(sel, v)

    @classmethod
    def m(cls, f: BoolArray, c: "ChoiceMap") -> "ChoiceMap":
        return choice_map_masked(f, c)

    @classmethod
    def a(cls, addr: Address, v: Any) -> "ChoiceMap":
        head = Selection.get_address_head(addr)
        tail = Selection.get_address_tail(addr)
        c = v if isinstance(v, ChoiceMap) else ChoiceMap.v(v)
        if tail:
            c = ChoiceMap.a(tail, c)
        if isinstance(head, DynamicAddressComponent):
            return choice_map_with_dyn_addr(head, c)
        else:
            return choice_map_with_static_addr(head, c)

    # NOTE: this only allows dictionaries with static keys
    # a.k.a. strings -- not jax.arrays -- for now.
    def addr_fn(self, addr_fn: dict):
        return choice_map_address_function(addr_fn, self)

    def into(self, addr: Address) -> "ChoiceMap":
        return ChoiceMap.a(addr, self)

    @classmethod
    def with_info(cls, info: String):
        return lambda fn: ChoiceMap(fn, info)

    ###################
    # Pretty printing #
    ###################

    def treescope_color(self):
        return pz.color_from_string(str(self.info))


ChoiceMapFunction = Callable[
    [AddressComponent],
    Tuple[Any, Bool, ChoiceMap],
    # The elements of this type are, given an address component:
    # 1. Value at this node, if there is one
    # 2. Did this choice map have this address component?
    # 3. The sub choice map at this node, if there is one
]


@ChoiceMap.with_info("Empty")
@Pytree.partial()
def choice_map_empty(head: AddressComponent):
    return None, False, choice_map_empty


@typecheck
def choice_map_value(v: Any):
    @ChoiceMap.with_info("Value")
    @Pytree.partial(v)
    def inner(v, head: AddressComponent) -> Any:
        match head:
            case ():
                return v, True, choice_map_empty
            case _:
                return None, False, choice_map_empty

    return inner


@typecheck
def choice_map_with_dyn_addr(addr: AddressComponent, c: ChoiceMap):
    @ChoiceMap.with_info(f"Dynamic(... => {c.info})")
    @Pytree.partial(addr, c)
    def inner(addr, c, head: AddressComponent):
        check = addr == head if isinstance(head, DynamicAddressComponent) else False
        return None, check, choice_map_masked(check, c)

    return inner


@typecheck
def choice_map_with_static_addr(addr: AddressComponent, c: ChoiceMap):
    @ChoiceMap.with_info(f"Static({addr} => {c.info})")
    @Pytree.partial(c)
    def inner(c, head: AddressComponent):
        check = addr == head
        return None, addr == head, choice_map_masked(check, c)

    return inner


def _extract_keys(data, search_value):
    new_dict = {}
    for key, value in data.items():
        if isinstance(key, tuple) and key and key[0] == search_value:
            # Create a new key from the remaining elements of the original key
            new_key = key[1:]
            # If the new key is a single-element tuple, convert it to just the element
            if len(new_key) == 1:
                new_key = new_key[0]
            new_dict[new_key] = value

    return new_dict


@typecheck
def choice_map_address_function(addr_fn: dict, c: ChoiceMap):
    @ChoiceMap.with_info(f"AddressFunction({addr_fn}, {c.info})")
    @Pytree.partial(c)
    def inner(c, head: StaticAddressComponent):
        sub_fn = _extract_keys(addr_fn, head)
        if sub_fn:
            return None, True, choice_map_address_function(sub_fn, c)
        else:
            new_head = addr_fn.get(head, head)
            if head == ():
                return c.call_recurse(new_head)
            elif isinstance(new_head, tuple):
                return c.call_recurse(new_head)
            else:
                return c.call(new_head)

    return inner


@typecheck
def choice_map_selection(sel: Selection, v: Any):
    @ChoiceMap.with_info(f"Selected({sel.info}) => {v}")
    @Pytree.partial(sel, v)
    def inner(sel, v, head: AddressComponent):
        if head == ():
            check = head in sel
            return (
                Mask.maybe_none(check * jnp.ones_like(v, dtype=bool), v),
                check,
                choice_map_selection(sel, v),
            )
        else:
            remaining = sel.step(head)
            v_arr = v
            if isinstance(head, DynamicAddressComponent):
                v_arr = jnp.array(v, copy=False)
                v_arr = v_arr[head] if v_arr.shape else v_arr
            return None, True, choice_map_selection(remaining, v_arr)

    return inner


@typecheck
def choice_map_xor(c1: ChoiceMap, c2: ChoiceMap):
    @ChoiceMap.with_info(f"({c1.info} ⊕ {c2.info})")
    @Pytree.partial(c1, c2)
    def inner(c1, c2, head: AddressComponent):
        match head:
            case ():
                v1, check1, _ = c1.call(head)
                v2, check2, _ = c2.call(head)
                check = staged_or(check1, check2)
                err_check = staged_and(check1, check2)
                staged_err(
                    err_check,
                    "The disjoint union of two choice maps have a value with the same address.",
                )

                def pair_bool_to_idx(bool1, bool2):
                    return (1 * bool1 + 2 * bool2 - 3 * (bool1 & bool2)) - 1

                idx = pair_bool_to_idx(check1, check2)
                v = Sum.maybe_none_or_mask(idx, v1, v2)
                return v, check, choice_map_empty
            case _:
                remaining_1 = c1.get_submap(head)
                remaining_2 = c2.get_submap(head)
                check = staged_or(c1.has_submap(head), c2.has_submap(head))
                return None, check, choice_map_xor(remaining_1, remaining_2)

    return inner


@typecheck
def choice_map_or(c1: ChoiceMap, c2: ChoiceMap):
    @ChoiceMap.with_info(f"({c1.info} + {c2.info})")
    @Pytree.partial(c1, c2)
    def inner(c1, c2, head: AddressComponent):
        match head:
            case ():
                v1, check1, _ = c1.call(head)
                v2, check2, _ = c2.call(head)
                check = staged_or(check1, check2)

                def pair_bool_to_idx(first, second):
                    output = -1 + first + 2 * (~first & second)
                    return output

                idx = pair_bool_to_idx(check1, check2)
                v = Sum.maybe_none_or_mask(idx, v1, v2)
                return v, check, choice_map_empty
            case _:
                remaining_1 = c1.get_submap(head)
                remaining_2 = c2.get_submap(head)
                check = staged_or(c1.has_submap(head), c2.has_submap(head))
                return None, check, choice_map_or(remaining_1, remaining_2)

    return inner


def choice_map_masked(flag: Bool | BoolArray, c: ChoiceMap):
    if static_check_is_concrete(flag) and isinstance(flag, Bool):

        @ChoiceMap.with_info(f"StaticMasked({flag}, {c.info})")
        @Pytree.partial(c)
        def inner(c, head: AddressComponent):
            v, check, submap = c.call(head)
            and_check = staged_and(flag, check)
            if static_check_is_concrete(and_check):
                return_map = submap
            else:
                return_map = choice_map_masked(and_check, submap)
            return (Mask.maybe_none(and_check, v), and_check, return_map)

    else:

        @ChoiceMap.with_info(f"Masked({c.info})")
        @Pytree.partial(flag, c)
        def inner(flag, c, head: AddressComponent):
            v, check, submap = c.call(head)
            and_check = staged_and(flag, check)
            return_map = choice_map_masked(and_check, submap)
            return (Mask.maybe_none(and_check, v), and_check, return_map)

    return inner


@typecheck
def choice_map_filtered(selection: Selection, c: ChoiceMap):
    @ChoiceMap.with_info(f"Filtered({selection.info}, {c.info})")
    @Pytree.partial(selection, c)
    def inner(selection, c, head: AddressComponent):
        check = head in selection
        match head:
            case ():
                return Mask.maybe_none(check, c()), check, choice_map_empty
            case _:
                return (
                    None,
                    check,
                    choice_map_filtered(selection.step(head), c.get_submap(head)),
                )

    return inner


@typecheck
def select_choice_map(c: ChoiceMap):
    @Selection.with_info(f"from ChoiceMap({c.info})")
    @Pytree.partial(c)
    def inner(c, head: AddressComponent):
        check = head in c
        return check, Selection.m(check, select_choice_map(c.get_submap(head)))

    return inner


#####
# Various selection functions
#####


@Selection.with_info("All")
@Pytree.partial()
def select_all(_: AddressComponent):
    return True, select_all


@typecheck
def select_defer(
    flag: Union[Bool, BoolArray],
    s: Selection,
) -> Selection:
    @Selection.with_info(f"Defer {s.info}")
    @Pytree.partial(flag, s)
    @typecheck
    def inner(
        flag: Union[Bool, BoolArray],
        s: Selection,
        head: AddressComponent,
    ):
        ch, remaining = s.has_addr(head)
        check = staged_and(flag, ch)
        return check, select_defer(check, remaining)

    return inner


@typecheck
def select_complement(s: Selection) -> Selection:
    @Selection.with_info(f"~{s.info}")
    @Pytree.partial(s)
    @typecheck
    def inner(s: Selection, head: AddressComponent):
        ch, remaining = s.has_addr(head)
        check = staged_not(ch)
        return check, select_complement(select_defer(ch, remaining))

    return inner


select_none = select_complement(select_all)


@typecheck
def select_static(addressed: AddressComponent) -> Selection:
    @Selection.with_info(f"Static '{addressed}'")
    @Pytree.partial()
    @typecheck
    def inner(head: AddressComponent):
        check = addressed == head or isinstance(head, EllipsisType)
        return check, select_defer(check, select_all)

    return inner


@typecheck
def select_idx(sidx: DynamicAddressComponent) -> Selection:
    @Selection.with_info("Dynamic")
    @Pytree.partial(sidx)
    @typecheck
    def inner(sidx: DynamicAddressComponent, head: AddressComponent):
        if isinstance(head, tuple) and not head:
            return False, select_none

        # If the head is either an ellipsis, or a static address component,
        # short circuit...
        ellipsis_check = isinstance(head, EllipsisType)
        if ellipsis_check:
            return (True, select_defer(True, select_all))

        if not isinstance(head, DynamicAddressComponent):
            return False, select_defer(False, select_all)

        # Else, we have some array comparison logic which is compatible with batching...
        def check_fn(v):
            check = staged_and(
                v,
                staged_or(
                    jnp.any(v == sidx),
                    ellipsis_check,
                ),
            )
            return check, select_defer(check, select_all)

        return (
            jax.vmap(check_fn)(head)
            if (not isinstance(head, Int) and head.shape)
            else check_fn(head)
        )

    return inner


@typecheck
def select_slice(slc: slice) -> Selection:
    @Selection.with_info(f"Slice {slc}")
    @Pytree.partial()
    @typecheck
    def inner(head: AddressComponent):
        # head is IntArray with shape = ().
        # head has type jnp.array with dtype = int, shape = ().
        # slc is a statically known slice.
        # we need to check
        # slice := slice(lower, upper, step)
        lower, upper, step = slc[0], slc[1], slc[2]
        check1 = head >= lower if lower else True
        check2 = head <= upper if upper else True

        # TODO: possibly doesn't work.
        check3 = head % step == lower if lower else True

        # No need to change.
        check = staged_and(staged_and(check1, check2), check3)
        return check, select_defer(check, select_all)

    return inner


@typecheck
def select_and(s1: Selection, s2: Selection) -> Selection:
    @Selection.with_info(f"{s1.info} & {s2.info}")
    @Pytree.partial(s1, s2)
    @typecheck
    def inner(s1: Selection, s2: Selection, head: AddressComponent):
        check1, remaining1 = s1.has_addr(head)
        check2, remaining2 = s2.has_addr(head)
        check = staged_and(check1, check2)
        return check, select_defer(check, select_and(remaining1, remaining2))

    return inner


@typecheck
def select_or(s1: Selection, s2: Selection) -> Selection:
    @Selection.with_info(f"{s1.info} | {s2.info}")
    @Pytree.partial(s1, s2)
    @typecheck
    def inner(s1: Selection, s2: Selection, head: AddressComponent):
        check1, remaining1 = s1.has_addr(head)
        check2, remaining2 = s2.has_addr(head)
        check = staged_or(check1, check2)
        return check, select_or(
            select_defer(check1, remaining1),
            select_defer(check2, remaining2),
        )

    return inner


@typecheck
def select_and_then(s1: Selection, s2: Selection) -> Selection:
    @Selection.with_info(f"({s1.info} >> {s2.info})")
    @Pytree.partial(s1, s2)
    @typecheck
    def inner(s1: Selection, s2: Selection, head: AddressComponent):
        check1, remaining1 = s1.has_addr(head)
        remaining = select_defer(check1, select_and(remaining1, s2))
        return check1, remaining

    return inner


##################################
# Custom choice map update specs #
##################################


@Pytree.dataclass(match_args=True)
class RemoveSelectionUpdateSpec(UpdateSpec):
    selection: Selection
