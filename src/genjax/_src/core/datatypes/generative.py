# Copyright 2023 MIT Probabilistic Computing Project
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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import rich.tree as rich_tree
from equinox import tree_at
from jax.experimental import checkify

import genjax._src.core.pretty_printing as gpp
from genjax._src.checkify import optional_check
from genjax._src.core.datatypes.hashable_dict import hashable_dict
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    BoolArray,
    Callable,
    FloatArray,
    IntArray,
    List,
    PRNGKey,
    Tuple,
    dispatch,
    typecheck,
)

#############
# Selection #
#############


class Selection(Pytree):
    def complement(self) -> "Selection":
        """Return a `Selection` which filters addresses to the complement set of the
        provided `Selection`.

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
            chm = tr.strip()
            selection = genjax.select("x")
            # ISSUE [#851](https://github.com/probcomp/genjax/issues/851): complement() is not implemented yet
            # complement = selection.complement()
            # filtered = chm.filter(complement)
            # print(console.render(filtered))
            ```
        """
        return ComplementSelection(self)


class MapSelection(Selection):
    @abstractmethod
    def get_subselection(self, addr) -> "Selection":
        raise NotImplementedError

    @abstractmethod
    def has_addr(self, addr) -> BoolArray:
        raise NotImplementedError

    ###########
    # Dunders #
    ###########

    def __getitem__(self, addr):
        subselection = self.get_subselection(addr)
        return subselection


class ComplementSelection(Selection):
    selection: Selection

    def complement(self):
        return self.selection

    def has_addr(self, addr):
        assert isinstance(self.selection, MapSelection)
        return jnp.logical_not(self.selection.has_addr(addr))

    def get_subselection(self, addr):
        assert isinstance(self.selection, MapSelection)
        return self.selection.get_subselection(addr).complement()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](Complement)")
        tree.add(self.selection.__rich_tree__())
        return tree


#######################
# Concrete selections #
#######################


class NoneSelection(Selection):
    def complement(self):
        return AllSelection()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](NoneSelection)")
        return tree


class AllSelection(Selection):
    def complement(self):
        return NoneSelection()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        return rich_tree.Tree("[bold](AllSelection)")


###########################
# Concrete map selections #
###########################


class HierarchicalSelection(MapSelection):
    trie: Trie

    @classmethod
    def from_addresses(cls, *addresses: Any):
        trie = Trie()
        for addr in addresses:
            trie = trie.trie_insert(addr, AllSelection())
        return HierarchicalSelection(trie)

    def has_addr(self, addr):
        return self.trie.has_submap(addr)

    def get_subselection(self, addr):
        value = self.trie.get_submap(addr)
        if value is None:
            return NoneSelection()
        else:
            subselect = value
            if isinstance(subselect, Trie):
                return HierarchicalSelection(subselect)
            else:
                return subselect

    # Extra method which is useful to generate an iterator
    # over keys and subselections at the first level.
    def get_subselections_shallow(self):
        def _inner(v):
            addr = v[0]
            submap = v[1].get_selection()
            if isinstance(submap, Trie):
                submap = HierarchicalSelection(submap)
            return (addr, submap)

        return map(
            _inner,
            self.trie.get_submaps_shallow(),
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](HierarchicalSelection)")
        for k, v in self.get_subselections_shallow():
            subk = tree.add(f"[bold]:{k}")
            subk.add(v.__rich_tree__())
        return tree


###########
# Choices #
###########


class Choice(Pytree):
    """`Choice` is the abstract base class of the type of random choices.

    The type `Choice` denotes a value which can be sampled from a generative function. There are many instances of `Choice` - distributions, for instance, utilize `ChoiceValue` - an implementor of `Choice` which wraps a single value. Other generative functions use map-like (or dictionary-like) `ChoiceMap` instances to represent their choices.
    """

    @abstractmethod
    def filter(self, selection: Selection) -> "Choice":
        pass

    @abstractmethod
    def merge(self, other: "Choice") -> Tuple["Choice", "Choice"]:
        pass

    @abstractmethod
    def get_selection(self) -> Selection:
        pass

    @abstractmethod
    def is_empty(self) -> BoolArray:
        pass

    @typecheck
    def partition(self, selection: Selection):
        return self.filter(selection), self.filter(selection.complement())

    @typecheck
    def surgery(self, selection: Selection, replacement: "Choice"):
        return tree_at(lambda v: v.filter(selection), replacement)

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

    def get_choice(self):
        return self

    def strip(self):
        return strip(self)


class EmptyChoice(Choice):
    """A `Choice` implementor which denotes an empty event."""

    def filter(self, selection):
        return self

    def is_empty(self):
        return jnp.array(True)

    def get_selection(self):
        return NoneSelection()

    @dispatch
    def merge(self, other):
        return other, self

    def __rich_tree__(self):
        return rich_tree.Tree("[bold](EmptyChoice)")


class ChoiceValue(Choice):
    value: Any

    def get_value(self):
        return self.value

    def is_empty(self):
        return jnp.array(False)

    @dispatch
    def merge(self, other: "ChoiceValue"):
        return self, other

    @dispatch
    def filter(self, selection: AllSelection):
        return self

    @dispatch
    def filter(self, selection):
        return EmptyChoice()

    def get_selection(self):
        return AllSelection()

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](ValueChoice)")
        tree.add(gpp.tree_pformat(self.value))
        return tree


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
    def get_submap(self, addr) -> Choice:
        pass

    @abstractmethod
    def has_submap(self, addr) -> BoolArray:
        pass

    ##############################################
    # Dispatch overloads for `Choice` interfaces #
    ##############################################

    @dispatch
    def filter(
        self,
        selection: AllSelection,
    ) -> "ChoiceMap":
        return self

    @dispatch
    def filter(
        self,
        selection: NoneSelection,
    ) -> EmptyChoice:
        return EmptyChoice()

    @dispatch
    def filter(
        self,
        selection: Selection,
    ) -> Choice:
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
            chm = tr.strip()
            selection = genjax.select("x")
            filtered = chm.filter(selection)
            print(console.render(filtered))
            ```
        """
        raise NotImplementedError

    def get_selection(self) -> "Selection":
        """Convert a `ChoiceMap` to a `Selection`."""
        raise Exception(
            f"`get_selection` is not implemented for choice map of type {type(self)}",
        )

    ###########
    # Dunders #
    ###########

    def __eq__(self, other):
        return self.tree_flatten() == other.tree_flatten()

    def __or__(self, other):
        return DisjointUnionChoiceMap([self, other])

    def __and__(self, other):
        return self.safe_merge(other)

    def __getitem__(self, addr: Any):
        if isinstance(addr, tuple):
            submap = self.get_submap(addr)
            if isinstance(submap, ChoiceValue):
                return submap.get_value()
            elif isinstance(submap, Mask):
                if isinstance(submap.value, ChoiceValue):
                    return submap.get_value()
                else:
                    return submap
            else:
                return submap
        else:
            return self.__getitem__((addr,))


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
            chm = tr.get_choice()
            v = chm.get_value()
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
    def get_choice(self) -> ChoiceMap:
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
            chm = tr.get_choice()
            print(console.render(chm))
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

    @dispatch
    def project(
        self,
        selection: NoneSelection,
    ) -> FloatArray:
        return 0.0

    @dispatch
    def project(
        self,
        selection: AllSelection,
    ) -> FloatArray:
        return self.get_score()

    @dispatch
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
        return self.get_choice()[x]


# Remove all trace metadata, and just return choices.
def strip(v):
    def _check(v):
        return isinstance(v, Trace)

    def _inner(v):
        if isinstance(v, Trace):
            return v.strip()
        else:
            return v

    return jtu.tree_map(_inner, v.get_choice(), is_leaf=_check)


###########
# Masking #
###########


class Mask(Choice):
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
        assert isinstance(self.value, ChoiceMap)
        return jnp.logical_and(self.flag, self.value.is_empty())

    def filter(self, selection: Selection):
        choices = self.value.get_choice()
        assert isinstance(choices, Choice)
        return Mask(self.flag, choices.filter(selection))

    def merge(self, other: Choice) -> Tuple["Mask", "Mask"]:
        pass

    def get_selection(self) -> Selection:
        # If a user chooses to `get_selection`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.flag)
            checkify.check(
                check_flag,
                "Attempted to convert a Mask to a Selection when the mask flag is False, meaning the masked value is invalid.\n",
            )

        optional_check(_check)
        if isinstance(self.value, Choice):
            return self.value.get_selection()
        else:
            return AllSelection()

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

    def get_submap(self, addr) -> Choice:
        # Using a `ChoiceMap` interface on the `Mask` means
        # that the value should be a `ChoiceMap`.
        assert isinstance(self.value, ChoiceMap)
        inner = self.value.get_submap(addr)
        if isinstance(inner, EmptyChoice):
            return inner
        else:
            return Mask(self.flag, inner)

    def has_submap(self, addr) -> BoolArray:
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
            chm: the choice map assignment ($c$)
            s: the score ($s$) of the choice map assignment
            retval: the return value from the generative function invocation

        Examples:
            Here's an example using a `genjax` distribution (`normal`). Distributions are generative functions, so they support the interface.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax

            console = genjax.console()

            key = jax.random.PRNGKey(314159)
            (chm, w, r) = genjax.normal.propose(key, (0.0, 1.0))
            print(console.render(chm))
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
            (chm, w, r) = model.propose(key, ())
            print(console.render(chm))
            ```
        """
        tr = self.simulate(key, args)
        chm = tr.get_choice()
        score = tr.get_score()
        retval = tr.get_retval()
        return (chm, score, retval)

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
            chm: A choice map indicating constraints ($u$).
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

        def _active(chm):
            tr, w = self.importance(key, chm, args)
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
        raise NotImplementedError

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

        def _some(chm):
            (new_tr, w, retdiff, _) = self.update(key, prev, chm, argdiffs)
            # The true_discards should match the Pytree type of possible_discards,
            # but these are valid.
            discard = Mask(True, possible_discards)
            primal = Diff.tree_primal(retdiff)
            retdiff = Diff.tree_diff_unknown_change(primal)
            return (new_tr, w, retdiff, discard)

        return new_constraints.match(_none, _some)

    def assess(
        self: "GenerativeFunction",
        chm: Choice,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        """Given a complete choice map indicating constraints ($u$) for all choices, and
        arguments ($x$), execute the generative function, and return the return value of
        the invocation, and the score of the choice map ($s$).

        Arguments:
            chm: A complete choice map indicating constraints ($u$) for all choices.
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
        chm = trace.strip().filter(selection)
        scorer, _ = self.unzip(key, fixed)
        grad, nograd = Pytree.tree_grad_split(
            (chm, trace.get_args()),
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

    @dispatch
    def filter(
        self,
        selection: MapSelection,
    ) -> Choice:
        trie = Trie()

        def _inner(k, v):
            sub = selection.get_subselection(k)
            under = v.filter(sub)
            return k, under

        iter = self.get_submaps_shallow()
        for k, v in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoice):
                trie = trie.trie_insert(k, v)

        if trie.is_static_empty():
            return EmptyChoice()

        return HierarchicalChoiceMap(trie)

    def has_submap(self, addr):
        return self.trie.has_submap(addr)

    def _lift_value(self, value):
        if value is None:
            return EmptyChoice()
        else:
            if isinstance(value, Trie):
                return HierarchicalChoiceMap(value)
            else:
                return value

    @dispatch
    def get_submap(self, addr: Any):
        value = self.trie.get_submap(addr)
        return self._lift_value(value)

    @dispatch
    def get_submap(self, addr: IntArray):
        value = self.trie.get_submap(addr)
        return self._lift_value(value)

    @dispatch
    def get_submap(self, addr: Tuple):
        first, *rest = addr
        top = self.get_submap(first)
        if isinstance(top, EmptyChoice):
            return top
        else:
            if rest:
                if len(rest) == 1:
                    rest = rest[0]
                else:
                    rest = tuple(rest)
                return top.get_submap(rest)
            else:
                return top

    def get_submaps_shallow(self):
        def _inner(v):
            addr = v[0]
            submap = v[1]
            if isinstance(submap, Trie):
                submap = HierarchicalChoiceMap(submap)
            return (addr, submap)

        return map(
            _inner,
            self.trie.get_submaps_shallow(),
        )

    def get_selection(self):
        trie = Trie()
        for k, v in self.get_submaps_shallow():
            trie = trie.trie_insert(k, v.get_selection())
        return HierarchicalSelection(trie)

    @dispatch
    def merge(self, other: "HierarchicalChoiceMap"):
        new = hashable_dict()
        discard = hashable_dict()
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


class DisjointUnionChoiceMap(ChoiceMap):
    """> A choice map combinator type which represents a disjoint union over multiple
    choice maps.

    The internal data representation of a `ChoiceMap` is often specialized to support optimized code generation for inference interfaces, but the address hierarchy which a `ChoiceMap` represents (as an assignment of choices to addresses) must be generic.

    To make this more concrete, a `VectorChoiceMap` represents choices with addresses of the form `(integer_index, ...)` - but its internal data representation is a struct-of-arrays. A `HierarchicalChoiceMap` can also represent address assignments with form `(integer_index, ...)` - but supporting choice map interfaces like `merge` across choice map types with specialized internal representations is complicated.

    Modeling languages might also make use of specialized representations for (JAX compatible) address uncertainty -- and addresses can contain runtime data e.g. `static` generative functions can support addresses `(dynamic_integer_index, ...)` where the index is not known at tracing time. When generative functions mix `(static_integer_index, ...)` and `(dynamic_integer_index, ...)` - resulting choice maps must be a type of disjoint union, whose methods include branching decisions on runtime data.

    To this end, `DisjointUnionChoiceMap` is a `ChoiceMap` type designed to support disjoint unions of choice maps of different types. It supports implementations of the choice map interfaces which are generic over the type of choice maps in the union, and also works with choice maps that contain runtime resolved address data.
    """

    submaps: List[ChoiceMap] = Pytree.field(default_factory=list)

    def has_submap(self, addr):
        checks = jnp.array(map(lambda v: v.has_submap(addr), self.submaps))
        return jnp.sum(checks) == 1

    def get_submap(self, head, *tail):
        new_submaps = list(
            filter(
                lambda v: not isinstance(v, EmptyChoice),
                map(lambda v: v.get_submap(head, *tail), self.submaps),
            )
        )
        # Static check: if any of the submaps are `ChoiceValue` instances, we must
        # check that all of them are. Otherwise, the choice map is invalid.
        check_address_leaves = list(
            map(lambda v: isinstance(v, ChoiceValue), new_submaps)
        )
        if any(check_address_leaves):
            assert all(map(lambda v: isinstance(v, ChoiceValue), new_submaps))

        if len(new_submaps) == 0:
            return EmptyChoice()
        elif len(new_submaps) == 1:
            return new_submaps[0]
        else:
            return DisjointUnionChoiceMap(new_submaps)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](DisjointUnionChoiceMap)")
        for submap in self.submaps:
            sub_tree = submap.__rich_tree__()
            tree.add(sub_tree)
        return tree
