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
from functools import reduce
from operator import or_

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import rich.tree as rich_tree
from jax import vmap
from jax.experimental import checkify

import genjax._src.core.pretty_printing as gpp
from genjax._src.checkify import optional_check
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import (
    staged_and,
    staged_check,
    staged_not,
    staged_or,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Address,
    AddressComponent,
    Any,
    Bool,
    BoolArray,
    Callable,
    DynamicAddressComponent,
    EllipsisType,
    FloatArray,
    Int,
    PRNGKey,
    StaticAddressComponent,
    String,
    Tuple,
    Union,
    static_check_is_concrete,
    typecheck,
)

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


class Selection(Pytree):
    has_addr: SelectionFunction
    level_info: String = Pytree.static()

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
        return f"Selection({self.level_info})"

    def __repr__(self):
        return f"Selection({self.level_info})"

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
    def m(cls, flag: BoolArray, s: "Selection") -> "Selection":
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
    def at(cls) -> SelectionBuilder:
        return Selection.SelectionBuilder()


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
    @Selection.with_info(f"Defer {s.level_info}")
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
    @Selection.with_info(f"~{s.level_info}")
    @Pytree.partial(s)
    @typecheck
    def inner(s: Selection, head: AddressComponent):
        ch, remaining = s.has_addr(head)
        check = staged_not(ch)
        return check, select_defer(check, select_complement(remaining))

    return inner


select_none = select_complement(select_all)


@typecheck
def select_static(addressed: AddressComponent) -> Selection:
    @Selection.with_info(f"Static {addressed}")
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
    @Selection.with_info(f"{s1.level_info} & {s2.level_info}")
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
    @Selection.with_info(f"{s1.level_info} | {s2.level_info}")
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
    @Selection.with_info(f"({s1.level_info} >> {s2.level_info})")
    @Pytree.partial(s1, s2)
    @typecheck
    def inner(s1: Selection, s2: Selection, head: AddressComponent):
        check1, remaining1 = s1.has_addr(head)
        remaining = select_defer(check1, select_and(remaining1, s2))
        return check1, remaining

    return inner


###########
# Choices #
###########


class Choice(Pytree):
    """`Choice` is the abstract base class of the type of random choices.

    The type `Choice` denotes a value which can be sampled from a generative function. There are many instances of `Choice` - distributions, for instance, utilize `ChoiceValue` - an implementor of `Choice` which wraps a single value. Other generative functions use map-like (or dictionary-like) `ChoiceMap` instances to represent their choices.
    """

    @abstractmethod
    def merge(self, other: "Choice") -> "Choice":
        pass

    @abstractmethod
    def is_empty(self) -> BoolArray:
        pass

    def get_choices(self):
        return self

    def strip(self):
        return strip(self)


def safe_merge(self, other: "Choice") -> "Choice":
    return self.safe_merge(other)


class ChoiceMap(Choice):
    """
    The type `ChoiceMap` denotes a map-like value which can be sampled from a generative function.

    Generative functions which utilize map-like representations often support a notion of _addressing_,
    allowing the invocation of generative function callees, whose choices become addressed random choices
    in the caller's choice map.
    """

    #######################
    # Map-like interfaces #
    #######################

    @abstractmethod
    def get_submap(self, addr: AddressComponent) -> "ChoiceMap":
        pass

    @abstractmethod
    def has_submap(self, addr: AddressComponent) -> BoolArray:
        pass

    @abstractmethod
    def filter(
        self,
        selection: Selection,
    ) -> "ChoiceMap":
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
        raise NotImplementedError

    @abstractmethod
    def get_selection(self) -> Selection:
        """Convert a `ChoiceMap` to a `Selection`."""
        raise NotImplementedError

    ###################
    # AddressIndex interface #
    ###################

    class AddressIndex(Pytree):
        choice_map: "ChoiceMap"
        selection: Selection

        def __getitem__(self, addr: Address) -> "ChoiceMap.AddressIndex":
            sel = Selection.at[addr]
            return ChoiceMap.AddressIndex(self.choice_map, sel | self.selection)

        def set(self, v):
            return self.choice_map + SelectionChoiceMap(self.selection, v)

        @property
        def also(self) -> "ChoiceMap.AddressIndex":
            return self

        def filter(self):
            return self.choice_map.filter(self.selection)

    @property
    def at(self) -> AddressIndex:
        if self is ChoiceMap:
            return ChoiceMap.AddressIndex(EmptyChoice(), Selection.n)
        else:
            return ChoiceMap.AddressIndex(self, Selection.n)

    ###########
    # Dunders #
    ###########

    def __add__(self, other):
        return self.merge(other)

    def __getitem__(self, addr: Address):
        addr = Pytree.tree_unwrap_const(addr)
        head = Selection.get_address_head(addr)
        tail = Selection.get_address_tail(addr)
        submap = self.get_submap(head)
        if tail:
            return submap[tail]
        else:
            return submap.get_submap(())

    def __contains__(self, addr: Address):
        addr = Pytree.tree_unwrap_const(addr)
        head = Selection.get_address_head(addr)
        tail = Selection.get_address_tail(addr)
        check = self.has_submap(head)
        submap = self.get_submap(head)
        if not tail:
            return submap.has_submap(tail)
        return staged_and(check, tail in submap)

    ################
    # Construction #
    ################

    @classproperty
    def n(cls) -> "ChoiceMap":
        return EmptyChoice()

    @classmethod
    def v(cls, v) -> "ChoiceMap":
        return ChoiceValue(v)


class EmptyChoice(ChoiceMap):
    """A `Choice` implementor which denotes an empty event."""

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        return self

    def has_submap(self, addr: AddressComponent) -> BoolArray:
        return False

    def filter(self, selection: Selection):
        return self

    def is_empty(self):
        return True

    def get_selection(self) -> Selection:
        return Selection.n

    def merge(self, other):
        return other

    def __rich_tree__(self):
        return rich_tree.Tree("[bold](EmptyChoice)")


class ValueLike(Choice):
    @abstractmethod
    def has_value(self) -> BoolArray:
        raise NotImplementedError

    @abstractmethod
    def get_value(self) -> Any:
        raise NotImplementedError


class ChoiceValue(ValueLike, ChoiceMap):
    value: Any

    def has_value(self):
        return True

    def get_value(self):
        return self.value

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        if addr == ():
            return self
        else:
            raise Exception("ChoiceValue doesn't address any choices.")

    def has_submap(self, addr: AddressComponent) -> BoolArray:
        return addr == ()

    def is_empty(self):
        return False

    def merge(self, other: Choice):
        check = other.is_empty()
        if static_check_is_concrete(check) and check:
            return self
        else:
            raise Exception("Cannot merge a ChoiceValue with a non-empty choice.")

    def filter(self, selection: Selection):
        check = selection[()]
        if static_check_is_concrete(check):
            if check:
                return self
            else:
                return EmptyChoice()
        else:
            return Mask.maybe(check, self)

    def get_selection(self) -> Selection:
        return Selection.a

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](ChoiceValue)")
        tree.add(gpp.tree_pformat(self.value))
        return tree


class DisjointUnionChoiceMap(ValueLike, ChoiceMap):
    first: ChoiceMap
    second: ChoiceMap

    @classmethod
    def maybe(cls, first, second):
        if staged_check(first.is_empty()):
            return second
        elif staged_check(second.is_empty()):
            return first
        else:
            return DisjointUnionChoiceMap(first, second)

    def is_empty(self) -> BoolArray:
        return staged_and(
            self.first.is_empty(),
            self.second.is_empty(),
        )

    def merge(self, other):
        return DisjointUnionChoiceMap.maybe(
            self.first.merge(other),
            self.second.merge(other),
        )

    def has_value(self) -> BoolArray:
        return staged_or(
            self.first.has_value(),
            self.second.has_value(),
        )

    def get_value(self) -> Any:
        val1 = self.first.get_value()
        val2 = self.second.get_value()
        return val1.merge(val2)

    def filter(self, selection: Selection) -> ChoiceMap:
        return DisjointUnionChoiceMap.maybe(
            self.first.filter(selection),
            self.second.filter(selection),
        )

    def has_submap(self, addr: Address) -> BoolArray:
        return staged_or(
            self.first.has_submap(addr),
            self.second.has_submap(addr),
        )

    def get_submap(self, addr: Address) -> ChoiceMap:
        return DisjointUnionChoiceMap.maybe(
            self.first.get_submap(addr),
            self.second.get_submap(addr),
        )

    def get_selection(self) -> Selection:
        return self.first.get_selection() | self.second.get_selection()


class SelectionChoiceMap(ValueLike, ChoiceMap):
    selection: Selection
    value: Any

    def is_empty(self) -> BoolArray:
        return staged_not(self.selection[...])

    def has_value(self) -> BoolArray:
        return self.selection[()]

    def get_value(self) -> Any:
        check = self.selection[()]
        if static_check_is_concrete(check):
            return (
                ChoiceValue(self.value)
                if isinstance(check, Bool)
                else Mask.maybe(check, ChoiceValue(self.value))
            )
        else:
            return Mask.maybe(check, self.value)

    def merge(self, other: ChoiceMap) -> ChoiceMap:
        return DisjointUnionChoiceMap(self, other)

    def get_selection(self) -> Selection:
        return self.selection

    def get_submap(self, addr: Address) -> ChoiceMap:
        if isinstance(addr, tuple) and addr == ():
            check = self.selection[()]
            return Mask.maybe(check, self.get_value())
        else:
            if isinstance(addr, DynamicAddressComponent):
                arr_addr = jnp.array(addr, copy=False)
                return (
                    jax.vmap(
                        lambda addr: SelectionChoiceMap(
                            self.selection.step(addr),
                            self.value[addr],
                        )
                    )(arr_addr)
                    if arr_addr.shape
                    else SelectionChoiceMap(
                        self.selection.step(arr_addr), self.value[arr_addr]
                    )
                )
            else:
                return SelectionChoiceMap(self.selection.step(addr), self.value)

    def has_submap(self, addr: Address) -> BoolArray:
        return self.selection.has(addr)

    def filter(self, selection: Selection) -> ChoiceMap:
        return SelectionChoiceMap(
            self.selection & selection,
            self.value,
        )

    def __str__(self):
        return f"SelectionChoiceMap(\n  selection: {self.selection.level_info}\n  value: {self.value}\n)"

    def __repr__(self):
        return f"SelectionChoiceMap(\n  selection: {self.selection.level_info}\n  value: {self.value}\n)"


#########
# Trace #
#########


class Trace(Pytree):
    """> Abstract base class for traces of generative functions.

    A `Trace` is a data structure used to represent sampled executions
    of generative functions.

    Traces track metadata associated with log probabilities of choices,
    as well as other data associated with the invocation of a generative
    function, including the arguments it was invoked with, its return
    value, and the identity of the generative function itself.
    """

    @abstractmethod
    def get_retval(self) -> Any:
        """Returns the return value from the generative function invocation which
        created the `Trace`.

        Examples:
            Here's an example using `genjax.normal` (a distribution). For distributions, the return value is the same as the (only) value in the returned choice map.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            retval = tr.get_retval()
            choice = tr.get_choices()
            v = choice.get_value()
            print(console.render((retval, v)))
            ```
        """

    @abstractmethod
    def get_score(self) -> FloatArray:
        """Return the score of the `Trace`.

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
            score = tr.get_score()
            x_score = bernoulli.logpdf(tr["x"], 0.3)
            y_score = bernoulli.logpdf(tr["y"], 0.3)
            print(console.render((score, x_score + y_score)))
            ```
        """

    @abstractmethod
    def get_args(self) -> Tuple:
        pass

    @abstractmethod
    def get_choices(self) -> ChoiceMap:
        """Return a `ChoiceMap` representation of the set of traced random choices
        sampled during the execution of the generative function to produce the `Trace`.

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
            choice = tr.get_choices()
            print(console.render(choice))
            ```
        """

    @abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        """Returns the generative function whose invocation created the `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            gen_fn = tr.get_gen_fn()
            print(console.render(gen_fn))
            ```
        """

    @abstractmethod
    def project(
        self,
        key: PRNGKey,
        selection: "Selection",
    ) -> FloatArray:
        """Given a `Selection`, return the total contribution to the score of the
        addresses contained within the `Selection`.

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
            selection = genjax.select("x")
            x_score = tr.project(key, selection)
            x_score_t = genjax.bernoulli.logpdf(tr["x"], 0.3)
            print(console.render((x_score_t, x_score)))
            ```
        """
        raise NotImplementedError

    def update(
        self,
        key: PRNGKey,
        choices: Choice,
        argdiffs: Tuple,
    ):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, choices, argdiffs)

    ##############################################
    # "In-place" update and projection interface #
    ##############################################

    class AddressIndex(Pytree):
        trace: "Trace"
        selection: Selection
        constraints: ChoiceMap

        @property
        def also(self) -> "Trace.AddressIndex":
            return self

        def set(self, v):
            choice_map = SelectionChoiceMap(self.selection, v)
            return Trace.AddressIndex(self.trace, Selection.a, choice_map)

        def update(self, key: PRNGKey, argdiffs=None):
            if argdiffs:
                return self.trace.update(key, self.constraints, argdiffs)
            else:
                return self.trace.update(
                    key,
                    self.constraints,
                    Diff.tree_diff_no_change(self.trace.get_args()),
                )

        def project(self, key: PRNGKey):
            return self.trace.project(key, self.selection)

        def __getitem__(self, addr: Address) -> FloatArray:
            selection = Selection.at[addr]
            return Trace.AddressIndex(
                self.trace, selection | self.selection, self.constraints
            )

    @property
    def at(self) -> AddressIndex:
        return Trace.AddressIndex(self, Selection.n, EmptyChoice())

    def get_aux(self) -> Tuple:
        raise NotImplementedError

    #############################
    # Default choice interfaces #
    #############################

    def is_empty(self):
        return self.strip().is_empty()

    def filter(
        self,
        selection: Selection,
    ) -> Any:
        stripped = self.strip()
        filtered = stripped.filter(selection)
        return filtered

    def merge(self, other: Choice) -> Tuple[Choice, Choice]:
        return self.strip().merge(other.strip())

    def get_selection(self):
        return self.strip().get_selection()

    def strip(self) -> Choice:
        return strip(self)

    def __getitem__(self, x):
        return self.get_choices()[x]


# Remove all trace metadata, and just return choices.
def strip(v):
    def _check(v):
        return isinstance(v, Trace)

    def inner(v):
        if isinstance(v, Trace):
            return v.strip()
        else:
            return v

    return jtu.tree_map(inner, v.get_choices(), is_leaf=_check)


###########
# Masking #
###########


# NOTE: we overload usage of `Mask` for masking choice maps, and for
# masking raw array/Python values.
class Mask(ValueLike, ChoiceMap):
    """The `Mask` choice datatype provides access to the masking system. The masking
    system is heavily influenced by the functional `Option` monad.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as `Choice` instances, and participate in inference computations (like scores, and importance weights or density ratios).

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    Users are expected to interact with `Mask` instances by either:

    * Unmasking them using the `Mask.unmask` interface. This interface uses JAX's `checkify` transformation to ensure that masked data exposed to a user is used only when valid. If a user chooses to `Mask.unmask` a `Mask` instance, they are also expected to use [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html) to transform their function to one which could return an error if the `Mask.flag` value is invalid.

    * Using `Mask.match` - which allows a user to provide "none" and "some" lambdas. The "none" lambda should accept no arguments, while the "some" lambda should accept an argument whose type is the same as the masked value. These lambdas should return the same type (`Pytree`, array, etc) of value.
    """

    flag: BoolArray
    value: Any

    # If the user provides a `Mask` as the value, we merge the flags and unwrap
    # one layer of the structure.
    def __post_init__(self):
        if isinstance(self.value, Mask):
            self.flag = staged_and(self.flag, self.value.flag)
            self.value = self.value.value

    @classmethod
    def maybe(cls, f: BoolArray, v: ChoiceMap):
        if isinstance(v, EmptyChoice):
            return v
        else:
            return (
                v
                if static_check_is_concrete(f) and isinstance(f, Bool) and f
                else EmptyChoice()
                if static_check_is_concrete(f) and isinstance(f, Bool)
                else Mask(f, v)
            )

    #####################
    # Choice interfaces #
    #####################

    def is_empty(self):
        check = staged_not(self.flag)
        if isinstance(self.value, Choice):
            return staged_and(check, self.value.is_empty())
        else:
            return check

    def merge(self, other: Choice) -> Choice:
        if static_check_is_concrete(self.flag) and self.flag:
            return self.value.merge(other)
        elif static_check_is_concrete(self.flag):
            return other
        else:
            return DisjointUnionChoiceMap(self, other)

    ###########################
    # Choice value interfaces #
    ###########################

    def has_value(self) -> BoolArray:
        return staged_and(self.flag, self.value.has_value())

    def get_value(self):
        # If a user chooses to `get_value`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.flag)
            checkify.check(
                check_flag,
                "Attempted to convert a Mask to a value when the mask flag is False, meaning the masked value is invalid.\n",
            )

        optional_check(_check)
        return self.value.get_value()

    #########################
    # Choice map interfaces #
    #########################

    def filter(self, selection: Selection) -> ChoiceMap:
        choices = self.value.get_choices()
        assert isinstance(choices, ChoiceMap)
        return Mask(self.flag, choices.filter(selection))

    def get_submap(self, addr: Address) -> ChoiceMap:
        # Using a `ChoiceMap` interface on the `Mask` means
        # that the value should be a `ChoiceMap`.
        assert isinstance(self.value, ChoiceMap)
        inner = self.value.get_submap(addr)
        return Mask.maybe(self.flag, inner)

    def has_submap(self, addr: Address) -> BoolArray:
        # Using a `ChoiceMap` interface on the `Mask` means
        # that the value should be a `ChoiceMap`.
        assert isinstance(self.value, ChoiceMap)
        inner_check = self.value.has_submap(addr)
        return staged_and(self.flag, inner_check)

    def get_selection(self) -> Selection:
        assert isinstance(self.value, ChoiceMap)
        return Selection.m(self.flag, self.value.get_selection())

    ######################
    # Masking interfaces #
    ######################

    def unmask(self):
        """> Unmask the `Mask`, returning the value within.

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` wraps valid data at runtime. To enforce validity checks, use the console context `genjax.console(enforce_checkify=True)` to handle any code which utilizes `Mask.unmask` with [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html).

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import genjax

            console = genjax.console()

            masked = genjax.Mask(True, jnp.ones(5))
            print(console.render(masked.unmask()))
            ```

            To enable runtime checks, the user must enable them explicitly in `genjax`.

            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.experimental.checkify as checkify
            import jax.numpy as jnp
            import genjax

            with genjax.console(enforce_checkify=True) as console:
                masked = genjax.Mask(False, jnp.ones(5))
                err, _ = checkify.checkify(masked.unmask)()
                print(console.render(err))
            ```
        """

        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.flag)
            checkify.check(
                check_flag,
                "Attempted to unmask when the mask flag is False: the masked value is invalid.\n",
            )

        optional_check(_check)
        return self.value

    def unsafe_unmask(self):
        # Unsafe version of unmask -- should only be used internally,
        # or carefully.
        return self.value

    @typecheck
    def match(self, some: Callable) -> Any:
        v = self.unmask()
        return some(v)

    @typecheck
    def safe_match(self, none: Callable, some: Callable) -> Any:
        return jax.lax.cond(
            self.flag,
            lambda v: some(v),
            lambda v: none(),
            self.value,
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        doc = gpp._pformat_array(self.flag, short_arrays=True)
        tree = rich_tree.Tree(f"[bold](Mask, {doc})")
        if isinstance(self.value, Pytree):
            val_tree = self.value.__rich_tree__()
            tree.add(val_tree)
        else:
            val_tree = gpp.tree_pformat(self.value, short_arrays=True)
            tree.add(val_tree)
        return tree


#######################
# Generative function #
#######################


class GenerativeFunction(Pytree):
    """> Abstract base class for generative functions.

    Generative functions are computational objects which expose convenient interfaces for probabilistic modeling and inference. They consist (often, subsets) of a few ingredients:

    * $p(c, r; x)$: a probability kernel over choice maps ($c$) and untraced randomness ($r$) given arguments ($x$).
    * $q(r; x, c)$: a probability kernel over untraced randomness ($r$) given arguments ($x$) and choice map assignments ($c$).
    * $f(x, c, r)$: a deterministic return value function.
    * $q(u; x, u')$: internal proposal distributions for choice map assignments ($u$) given other assignments ($u'$) and arguments ($x$).

    The interface of methods and associated datatypes which these objects expose is called _the generative function interface_ (GFI). Inference algorithms are written against this interface, providing a layer of abstraction above the implementation.

    Generative functions are allowed to partially implement the interface, with the consequence that partially implemented generative functions may have restricted inference behavior.

    !!! info "Interaction with JAX"

        Concrete implementations of `GenerativeFunction` will likely interact with the JAX tracing machinery if used with the languages exposed by `genjax`. Hence, there are specific implementation requirements which are more stringent than the requirements
        enforced in other Gen implementations (e.g. Gen in Julia).

        * For broad compatibility, the implementation of the interfaces *should* be compatible with JAX tracing.
        * If a user wishes to implement a generative function which is not compatible with JAX tracing, that generative function may invoke other JAX compat generative functions, but likely cannot be invoked inside of JAX compat generative functions.

    Aside from JAX compatibility, an implementor *should* match the interface signatures documented below. This is not statically checked - but failure to do so
    will lead to unintended behavior or errors.
    """

    def simulate(
        self: "GenerativeFunction",
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        """Given a `key: PRNGKey` and arguments `x: Tuple`, samples a choice map $c
        \\sim p(\\cdot; x)$, as well as any untraced randomness $r \\sim p(\\cdot; x,
        c)$ to produce a trace $t = (x, c, r)$.

        While the types of traces `t` are formally defined by $(x, c, r)$, they will often store additional information - like the _score_ ($s$):

        $$
        s = \\log \\frac{p(c, r; x)}{q(r; x, c)}
        $$

        Arguments:
            key: A `PRNGKey`.
            args: Arguments to the generative function.

        Returns:
            tr: A trace capturing the data and inference data associated with the generative function invocation.

        Examples:
            Here's an example using a `genjax` distribution (`normal`). Distributions are generative functions, so they support the interface.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            print(console.render(tr))
            ```

            Here's a slightly more complicated example using the `static` generative function language. You can find more examples on the `static` language page.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()


            @genjax.static_gen_fn
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                y = genjax.normal(x, 1.0) @ "y"
                return y


            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            print(console.render(tr))
            ```
        """
        raise NotImplementedError

    def propose(
        self: "GenerativeFunction",
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple[Choice, FloatArray, Any]:
        """Given a `key: PRNGKey` and arguments ($x$), execute the generative function,
        returning a tuple containing the return value from the generative function call,
        the score ($s$) of the choice map assignment, and the choice map ($c$).

        The default implementation just calls `simulate`, and then extracts the data from the `Trace` returned by `simulate`. Custom generative functions can overload the implementation for their own uses (e.g. if they don't have an associated `Trace` datatype, but can be used as a proposal).

        Arguments:
            key: A `PRNGKey`.
            args: Arguments to the generative function.

        Returns:
            choice: the choice map assignment ($c$)
            s: the score ($s$) of the choice map assignment
            retval: the return value from the generative function invocation

        Examples:
            Here's an example using a `genjax` distribution (`normal`). Distributions are generative functions, so they support the interface.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()

            key = jax.random.PRNGKey(314159)
            (choice, w, r) = genjax.normal.propose(key, (0.0, 1.0))
            print(console.render(choice))
            ```

            Here's a slightly more complicated example using the `static` generative function language. You can find more examples on the `static` language page.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()


            @genjax.static_gen_fn
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                y = genjax.normal(x, 1.0) @ "y"
                return y


            key = jax.random.PRNGKey(314159)
            (choice, w, r) = model.propose(key, ())
            print(console.render(choice))
            ```
        """
        tr = self.simulate(key, args)
        choice = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        return (choice, score, retval)

    def importance(
        self: "GenerativeFunction",
        key: PRNGKey,
        choice: Choice,
        args: Tuple,
    ) -> Tuple[Trace, FloatArray]:
        """Given a `key: PRNGKey`, a choice map indicating constraints ($u$), and
        arguments ($x$), execute the generative function, and return an importance
        weight estimate of the conditional density evaluated at the non-constrained
        choices, and a trace whose choice map ($c = u' ⧺ u$) is consistent with the
        constraints ($u$), with unconstrained choices ($u'$) proposed from an internal
        proposal.

        Arguments:
            key: A `PRNGKey`.
            choice: A choice map indicating constraints ($u$).
            args: Arguments to the generative function ($x$).

        Returns:
            tr: A trace capturing the data and inference data associated with the generative function invocation.
            w: An importance weight.

        The importance weight `w` is given by:

        $$
        w = \\log \\frac{p(u' ⧺ u, r; x)}{q(u'; u, x)q(r; x, t)}
        $$
        """
        raise NotImplementedError

    def update(
        self: "GenerativeFunction",
        key: PRNGKey,
        prev: Trace,
        new_constraints: Choice,
        diffs: Tuple,
    ) -> Tuple[Trace, FloatArray, Any, Choice]:
        primals = Diff.tree_primal(diffs)
        prev_choice = prev.get_choices()
        merged, discarded = prev_choice.merge(new_constraints)
        (tr, _) = self.importance(key, merged, primals)
        retval = tr.get_retval()
        return (tr, tr.get_score() - prev.get_score(), retval, discarded)

    def assess(
        self: "GenerativeFunction",
        choice: Choice,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        """Given a complete choice map indicating constraints ($u$) for all choices, and
        arguments ($x$), execute the generative function, and return the return value of
        the invocation, and the score of the choice map ($s$).

        Arguments:
            choice: A complete choice map indicating constraints ($u$) for all choices.
            args: Arguments to the generative function ($x$).

        Returns:
            score: The score of the choice map.
            retval: The return value from the generative function invocation.

        The score ($s$) is given by:

        $$
        s = \\log \\frac{p(c, r; x)}{q(r; x, c)}
        $$
        """
        raise NotImplementedError

    def sample_retval(self, key: PRNGKey, args: Tuple) -> Any:
        return self.simulate(key, args).get_retval()

    def restore_with_aux(
        self,
        interface_data: Tuple,
        aux: Tuple,
    ) -> Trace:
        raise NotImplementedError


class JAXGenerativeFunction(GenerativeFunction, Pytree):
    """A `GenerativeFunction` subclass for JAX compatible generative functions.

    Mixing in this class denotes that a generative function implementation can be used within a calling context where JAX transformations are being applied, or JAX tracing is being applied (e.g. `jax.jit`). As a callee in other generative functions, this type exposes an `__abstract_call__` method which can be use to customize the behavior under abstract tracing (a default is provided, and users are not expected to interact with this functionality).

    Compatibility with JAX tracing allows generative functions that mixin this class to expose several default methods which support convenient access to gradient computation using `jax.grad`.
    """

    @typecheck
    def unzip(
        self, fixed: Choice
    ) -> Tuple[
        Callable[[Choice, Tuple], FloatArray],
        Callable[[Choice, Tuple], Any],
    ]:
        """The `unzip` method expects a fixed (under gradients) `Choice` argument, and
        returns two `Callable` instances: the first exposes a.

        pure function from `(differentiable: Tuple, nondifferentiable: Tuple)
        -> score` where `score` is the log density returned by the `assess`
        method, and the second exposes a pure function from `(differentiable:
        Tuple, nondifferentiable: Tuple) -> retval` where `retval` is the
        returned value from the `assess` method.

        Arguments:
            fixed: A fixed choice map.
        """

        def score(differentiable: Tuple, nondifferentiable: Tuple) -> FloatArray:
            provided, args = Pytree.tree_grad_zip(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            (score, _) = self.assess(merged, args)
            return score

        def retval(differentiable: Tuple, nondifferentiable: Tuple) -> Any:
            provided, args = Pytree.tree_grad_zip(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            (_, retval) = self.assess(merged, args)
            return retval

        return score, retval

    # A higher-level gradient API - it relies upon `unzip`,
    # but provides convenient access to first-order gradients.
    @typecheck
    def choice_grad(self, key: PRNGKey, trace: Trace, selection: Selection):
        fixed = trace.strip().filter(~selection)
        choice = trace.strip().filter(selection)
        scorer, _ = self.unzip(key, fixed)
        grad, nograd = Pytree.tree_grad_split(
            (choice, trace.get_args()),
        )
        choice_gradient_tree, _ = jax.grad(scorer)(grad, nograd)
        return choice_gradient_tree

    def __abstract_call__(self, *args) -> Any:
        """Used to support JAX tracing, although this default implementation involves no
        JAX operations (it takes a fixed-key sample from the return value).

        Generative functions may customize this to improve compilation time.
        """
        return self.simulate(jax.random.PRNGKey(0), args).get_retval()
