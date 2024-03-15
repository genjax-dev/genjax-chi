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
from jax.experimental import checkify

import genjax._src.core.pretty_printing as gpp
from genjax._src.checkify import optional_check
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Address,
    AddressComponent,
    Any,
    Bool,
    BoolArray,
    Callable,
    DynamicAddressComponent,
    FloatArray,
    PRNGKey,
    StaticAddressComponent,
    Tuple,
    dispatch,
    static_check_is_concrete,
    typecheck,
)

##############
# Selections #
##############

SelectionFunction = Callable[[AddressComponent], Tuple[Bool, "Selection"]]


# NOTE: normal class properties are deprecated in Python 3.11,
# so here's our simple custom version.
class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


class Selection(Pytree):
    has_addr: SelectionFunction

    def __and__(self, other: "Selection") -> "Selection":
        return select_and(self, other)

    def __or__(self, other: "Selection") -> "Selection":
        return select_or(self, other)

    def __not__(self) -> "Selection":
        return select_complement(self)

    def __gt__(self, other):
        return select_and_then(self, other)

    def __lt__(self, other):
        return select_and_then(other, self)

    @classmethod
    def get_address_head(cls, addr: Address) -> AddressComponent:
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

    def both(self, addr: AddressComponent) -> Tuple[BoolArray, "Selection"]:
        return self.has_addr(addr)

    def do(self, addr: Address):
        head = Selection.get_address_head(addr)
        check1, remaining = self.both(head)
        if static_check_is_concrete(check1) and check1:
            tail = Selection.get_address_tail(addr)
            if tail:
                return remaining.do(tail)
            else:
                return True, select_all
        elif not static_check_is_concrete(check1):
            tail = Selection.get_address_tail(addr)
            if tail:
                masked_remaining = select_masked(check1, remaining)
                return masked_remaining.do(tail)
            else:
                return check1, select_all
        else:
            return check1, select_none

    def __call__(self, addr: Address) -> "Selection":
        _, remaining = self.do(addr)
        return remaining

    def __getitem__(self, addr: Address) -> BoolArray:
        check, _ = self.do(addr)
        return check

    def has(self, addr: Address):
        return self.__getitem__(addr)

    @classproperty
    def n(cls) -> "Selection":
        return select_none

    @classproperty
    def a(cls) -> "Selection":
        return select_all

    @classmethod
    def s(cls, addr) -> "Selection":
        return select_static(addr)

    @classmethod
    def idx(cls, addr) -> "Selection":
        return select_idx(addr)

    @classmethod
    def f(cls, *addrs) -> "Selection":
        return reduce(or_, (cls.s(addr) for addr in addrs))

    @classmethod
    def m(cls, flag: BoolArray, s: "Selection") -> "Selection":
        return select_masked(flag, s)

    @classmethod
    def reduce_or(cls, selections) -> "Selection":
        return reduce(or_, selections)


#####
# Various selection functions
#####


@Selection
@Pytree.const
def select_all(_: AddressComponent):
    return True, select_all


@Selection
@Pytree.const
def select_none(_: AddressComponent):
    return False, select_none


@typecheck
def select_static(addressed: AddressComponent) -> Selection:
    @Selection
    @Pytree.const
    @typecheck
    def inner(addr: AddressComponent):
        if addressed == addr:
            return True, select_all
        else:
            return False, select_none

    return inner


@typecheck
def select_idx(sidx: DynamicAddressComponent) -> Selection:
    @Selection
    @Pytree.partial(sidx)
    @typecheck
    def inner(sidx: DynamicAddressComponent, idx: DynamicAddressComponent):
        check = idx == sidx
        return jnp.any(check), select_all

    return inner


@typecheck
def select_and_then(s1: Selection, s2: Selection) -> Selection:
    @Selection
    @Pytree.const
    @typecheck
    def inner(addr_comp: AddressComponent):
        check1, remaining = s1.both(addr_comp)
        return check1, select_and(remaining, s2)

    return inner


@typecheck
def select_complement(s: Selection) -> Selection:
    @Selection
    @Pytree.const
    @typecheck
    def inner(addr_comp: AddressComponent):
        check, remaining = s.both(addr_comp)
        return jnp.logical_not(check), select_complement(remaining)

    return inner


@typecheck
def select_and(s1: Selection, s2: Selection) -> Selection:
    @Selection
    @Pytree.const
    @typecheck
    def inner(addr_comp: AddressComponent):
        check1, remaining1 = s1.both(addr_comp)
        check2, remaining2 = s2.both(addr_comp)
        return check1 and check2, select_and(remaining1, remaining2)

    return inner


@typecheck
def select_or(s1: Selection, s2: Selection) -> Selection:
    @Selection
    @Pytree.const
    @typecheck
    def inner(addr_comp: AddressComponent):
        check1, remaining1 = s1.both(addr_comp)
        check2, remaining2 = s2.both(addr_comp)
        return check1 or check2, select_or(remaining1, remaining2)

    return inner


@typecheck
def select_masked(flag: BoolArray, s: Selection) -> Selection:
    @Selection
    @Pytree.const
    @typecheck
    def inner(addr_comp: AddressComponent):
        check, remaining = s.both(addr_comp)
        check = jnp.logical_and(flag, check)
        return check, select_masked(flag, remaining)

    return inner


###########
# Choices #
###########


class Choice(Pytree):
    """`Choice` is the abstract base class of the type of random choices.

    The type `Choice` denotes a value which can be sampled from a generative function. There are many instances of `Choice` - distributions, for instance, utilize `ChoiceValue` - an implementor of `Choice` which wraps a single value. Other generative functions use map-like (or dictionary-like) `ChoiceMap` instances to represent their choices.
    """

    @abstractmethod
    def merge(self, other: "Choice") -> Tuple["Choice", "Choice"]:
        pass

    @abstractmethod
    def is_empty(self) -> BoolArray:
        pass

    def safe_merge(self, other: "Choice") -> "Choice":
        new, discard = self.merge(other)

        # If the discarded choice is not empty, raise an error.
        # However, it's possible that we don't know that the discarded
        # choice is empty until runtime, so we use checkify.
        def _check():
            check_flag = jnp.logical_not(discard.is_empty())
            checkify.check(
                check_flag,
                "The discarded choice is not empty.",
            )

        optional_check(_check)
        return new

    # This just ignores the discard check.
    def unsafe_merge(self, other: "Choice") -> "Choice":
        new, _ = self.merge(other)
        return new

    def get_choices(self):
        return self

    def strip(self):
        return strip(self)


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

    ##############################################
    # Dispatch overloads for `Choice` interfaces #
    ##############################################

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

    ###########
    # Dunders #
    ###########

    def __eq__(self, other):
        return self.tree_flatten() == other.tree_flatten()

    def __and__(self, other):
        return self.safe_merge(other)

    def do(self, addr: Address):
        head = Selection.get_address_head(addr)
        submap = self.get_submap(head)
        tail = Selection.get_address_tail(addr)
        return submap.do(tail)

    def __getitem__(self, addr: Address):
        return self.do(addr)


class EmptyChoice(ChoiceMap):
    """A `Choice` implementor which denotes an empty event."""

    def get_submap(self, addr: AddressComponent):
        return self

    def has_submap(self, addr: AddressComponent):
        return False

    def filter(self, selection: Selection):
        return self

    def is_empty(self):
        return jnp.array(True)

    def get_selection(self) -> Selection:
        return Selection.none

    def merge(self, other):
        return other, self

    def do(self, addr: Address):
        return self

    def __rich_tree__(self):
        return rich_tree.Tree("[bold](EmptyChoice)")


class ChoiceValue(ChoiceMap):
    value: Any

    def get_value(self):
        return self.value

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        raise Exception("ChoiceValue doesn't address any choices.")

    def has_submap(self, addr: AddressComponent) -> BoolArray:
        return False

    def is_empty(self):
        return jnp.array(False)

    def merge(self, other: Choice):
        return other, self

    def filter(self, selection: Selection):
        check = selection[...]
        if check:
            return self
        else:
            return EmptyChoice()

    def get_selection(self) -> Selection:
        return Selection.a

    def do(self, addr: Address):
        if addr:
            self.get_submap(addr)
        else:
            return self.value

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](ChoiceValue)")
        tree.add(gpp.tree_pformat(self.value))
        return tree


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
    def project(self, selection: "Selection") -> FloatArray:
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
            x_score = tr.project(selection)
            x_score_t = genjax.bernoulli.logpdf(tr["x"], 0.3)
            print(console.render((x_score_t, x_score)))
            ```
        """
        raise NotImplementedError

    @dispatch
    def update(
        self,
        key: PRNGKey,
        choices: Choice,
        argdiffs: Tuple,
    ):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, choices, argdiffs)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        choices: Choice,
    ):
        gen_fn = self.get_gen_fn()
        args = self.get_args()
        argdiffs = Diff.tree_diff_no_change(args)
        return gen_fn.update(key, self, choices, argdiffs)

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


class Mask(ChoiceMap):
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
            self.flag = jnp.logical_and(self.flag, self.value.flag)
            self.value = self.value.value

    #####################
    # Choice interfaces #
    #####################

    def is_empty(self):
        assert isinstance(self.value, Choice)
        return jnp.logical_and(self.flag, self.value.is_empty())

    def filter(self, selection: Selection):
        choices = self.value.get_choices()
        assert isinstance(choices, ChoiceMap)
        return Mask(self.flag, choices.filter(selection))

    def merge(self, other: Choice) -> Tuple[Choice, Choice]:
        pass

    def get_selection(self) -> Selection:
        assert isinstance(self.value, ChoiceMap)
        return Selection.m(self.flag, self.value.get_selection())

    ###########################
    # Choice value interfaces #
    ###########################

    def get_value(self):
        # Using a `ChoiceValue` interface on the `Mask` means
        # that the value should be a `ChoiceValue`.
        assert isinstance(self.value, ChoiceValue)

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

    def get_submap(self, addr: AddressComponent) -> ChoiceMap:
        # Using a `ChoiceMap` interface on the `Mask` means
        # that the value should be a `ChoiceMap`.
        assert isinstance(self.value, ChoiceMap)
        inner = self.value.get_submap(addr)
        if isinstance(inner, EmptyChoice):
            return inner
        else:
            return Mask(self.flag, inner)

    def has_submap(self, addr: AddressComponent) -> BoolArray:
        # Using a `ChoiceMap` interface on the `Mask` means
        # that the value should be a `ChoiceMap`.
        assert isinstance(self.value, ChoiceMap)
        inner_check = self.value.has_submap(addr)
        return jnp.logical_and(self.flag, inner_check)

    ######################
    # Masking interfaces #
    ######################

    @typecheck
    def match(self, none: Callable, some: Callable) -> Any:
        """> Pattern match on the `Mask` type - by providing "none"
        and "some" lambdas.

        The "none" lambda should accept no arguments, while the "some" lambda should accept the same type as the value in the `Mask`. Both lambdas should return the same type (array, or `jax.Pytree`).

        Arguments:
            none: A lambda to handle the "none" branch. The type of the return value must agree with the "some" branch.
            some: A lambda to handle the "some" branch. The type of the return value must agree with the "none" branch.

        Returns:
            value: A value computed by either the "none" or "some" lambda, depending on if the `Mask` is valid (e.g. `Mask.mask` is `True`).

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import genjax

            console = genjax.console()

            masked = genjax.Mask(False, jnp.ones(5))
            v1 = masked.match(lambda: 10.0, lambda v: jnp.sum(v))
            masked = genjax.Mask(True, jnp.ones(5))
            v2 = masked.match(lambda: 10.0, lambda v: jnp.sum(v))
            print(console.render((v1, v2)))
            ```
        """
        flag = jnp.array(self.flag)
        if flag.shape == ():
            return jax.lax.cond(
                flag,
                lambda: some(self.value),
                lambda: none(),
            )
        else:
            return jax.lax.select(
                flag,
                some(self.value),
                none(),
            )

    @typecheck
    def just_match(self, some: Callable) -> Any:
        v = self.unmask()
        return some(v)

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
        # Unsafe version of unmask -- should only be used internally.
        return self.value

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

    @dispatch
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

    @dispatch
    def importance(
        self: "GenerativeFunction",
        key: PRNGKey,
        constraints: Mask,
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
            constraints: A choice map indicating constraints ($u$).
            args: Arguments to the generative function ($x$).

        Returns:
            tr: A trace capturing the data and inference data associated with the generative function invocation.
            w: An importance weight.

        The importance weight `w` is given by:

        $$
        w = \\log \\frac{p(u' ⧺ u, r; x)}{q(u'; u, x)q(r; x, t)}
        $$
        """

        def _inactive():
            w = 0.0
            tr = self.simulate(key, args)
            return tr, w

        def _active(choice):
            tr, w = self.importance(key, choice, args)
            return tr, w

        return constraints.match(_inactive, _active)

    @dispatch
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

    @dispatch
    def update(
        self: "GenerativeFunction",
        key: PRNGKey,
        prev: Trace,
        new_constraints: Mask,
        argdiffs: Tuple,
    ) -> Tuple[Trace, FloatArray, Any, Mask]:
        # The semantics of the merge operation entail that the second returned value
        # is the discarded values after the merge.
        discard_option = prev.strip()
        possible_constraints = new_constraints.unsafe_unmask()
        _, possible_discards = discard_option.merge(possible_constraints)

        def _none():
            (new_tr, w, retdiff, _) = self.update(key, prev, EmptyChoice(), argdiffs)
            discard = Mask(False, possible_discards)
            primal = Diff.tree_primal(retdiff)
            retdiff = Diff.tree_diff_unknown_change(primal)
            return (new_tr, w, retdiff, discard)

        def _some(choice):
            (new_tr, w, retdiff, _) = self.update(key, prev, choice, argdiffs)
            # The true_discards should match the Pytree type of possible_discards,
            # but these are valid.
            discard = Mask(True, possible_discards)
            primal = Diff.tree_primal(retdiff)
            retdiff = Diff.tree_diff_unknown_change(primal)
            return (new_tr, w, retdiff, discard)

        return new_constraints.match(_none, _some)

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
        self,
        fixed: Choice,
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
        fixed = trace.strip().filter(selection.complement())
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


########################
# Concrete choice maps #
########################


class HierarchicalChoiceMap(ChoiceMap):
    trie: Trie = Pytree.field(default_factory=Trie)

    def is_empty(self) -> BoolArray:
        iter = self.get_submaps_shallow()
        check = jnp.array(True)
        for _, v in iter:
            check = jnp.logical_and(check, v.is_empty())
        return check

    def filter(
        self,
        selection: Selection,
    ) -> ChoiceMap:
        trie = Trie()

        def inner(k, v):
            check, remaining = selection.both(k)
            under = v.filter(remaining)
            return k, under

        iter = self.get_submaps_shallow()
        for k, v in map(lambda args: inner(*args), iter):
            if not isinstance(v, EmptyChoice):
                trie = trie.trie_insert(k, v)

        if trie.is_static_empty():
            return EmptyChoice()

        return HierarchicalChoiceMap(trie)

    @typecheck
    def has_submap(self, addr: StaticAddressComponent) -> BoolArray:
        return self.trie.has_submap(addr)

    def _lift_value(self, value):
        if value is None:
            return EmptyChoice()
        else:
            if isinstance(value, Trie):
                return HierarchicalChoiceMap(value)
            else:
                return value

    @typecheck
    def get_submap(self, addr: StaticAddressComponent) -> ChoiceMap:
        value = self.trie.get_submap(addr)
        return self._lift_value(value)

    def get_submaps_shallow(self):
        def inner(v):
            addr = v[0]
            submap = v[1]
            if isinstance(submap, Trie):
                submap = HierarchicalChoiceMap(submap)
            return (addr, submap)

        return map(
            inner,
            self.trie.get_submaps_shallow(),
        )

    def get_selection(self):
        return Selection.reduce_or(
            (Selection.s(k) > v.get_selection()) for k, v in self.get_submaps_shallow()
        )

    @dispatch
    def merge(self, other: "HierarchicalChoiceMap"):
        new = dict()
        discard = dict()
        for k, v in self.get_submaps_shallow():
            if other.has_submap(k):
                sub = other.get_submap(k)
                new[k], discard[k] = v.merge(sub)
            else:
                new[k] = v
        for k, v in other.get_submaps_shallow():
            if not self.has_submap(k):
                new[k] = v
        return HierarchicalChoiceMap(Trie(new)), HierarchicalChoiceMap(Trie(discard))

    @dispatch
    def merge(self, other: EmptyChoice):
        return self, other

    @dispatch
    def merge(self, other: ChoiceValue):
        return other, self

    @dispatch
    def merge(self, other: ChoiceMap):
        raise Exception(
            f"Merging with choice map type {type(other)} not supported.",
        )

    def insert(self, k, v):
        v = (
            ChoiceValue(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        return HierarchicalChoiceMap(self.trie.trie_insert(k, v))

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](HierarchicalChoiceMap)")
        for k, v in self.get_submaps_shallow():
            subk = rich_tree.Tree(f"[bold]:{k}")
            subv = v.__rich_tree__()
            subk.add(subv)
            tree.add(subk)
        return tree
