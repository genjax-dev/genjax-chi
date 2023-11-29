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

import abc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import rich.tree as rich_tree
from jax.experimental.checkify import checkify

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.hashable_dict import hashable_dict
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.incremental import tree_diff_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.interpreters.incremental import tree_diff_unknown_change
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.utilities import tree_grad_split
from genjax._src.core.pytree.utilities import tree_zipper
from genjax._src.core.typing import Any
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import static_check_is_concrete
from genjax._src.core.typing import typecheck
from genjax._src.global_options import global_options


#############
# Utilities #
#############


########################
# Generative datatypes #
########################


#############
# Selection #
#############


@dataclass
class Selection(Pytree):
    def complement(self) -> "Selection":
        """Return a `Selection` which filters addresses to the complement set
        of the provided `Selection`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen(genjax.Static)
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.strip()
            selection = genjax.select("x")
            complement = selection.complement()
            filtered = chm.filter(complement)
            print(console.render(filtered))
            ```
        """
        return ComplementSelection(self)

    @abc.abstractmethod
    def get_subselection(self, addr) -> "Selection":
        raise NotImplementedError

    @abc.abstractmethod
    def has_addr(self, addr) -> BoolArray:
        raise NotImplementedError

    ###########
    # Dunders #
    ###########

    def __getitem__(self, addr):
        subselection = self.get_subselection(addr)
        return subselection


@dataclass
class ComplementSelection(Selection):
    selection: Selection

    def flatten(self):
        return (self.selection,), ()

    def complement(self):
        return self.selection

    def has_addr(self, addr):
        return jnp.logical_not(self.selection.has_addr(addr))

    def get_subselection(self, addr):
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


@dataclass
class NoneSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return NoneSelection()

    def complement(self):
        return AllSelection()

    def has_addr(self, addr):
        return False

    def get_subselection(self, addr):
        return self

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](NoneSelection)")
        return tree


@dataclass
class AllSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return AllSelection()

    def complement(self):
        return NoneSelection()

    def has_addr(self, addr):
        return True

    def get_subselection(self, addr):
        return self

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        return rich_tree.Tree("[bold](AllSelection)")


##################################
# Concrete structured selections #
##################################


@dataclass
class HierarchicalSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @classmethod
    @dispatch
    def new(cls, *addrs: Any):
        trie = Trie.new()
        for addr in addrs:
            trie[addr] = AllSelection()
        return HierarchicalSelection(trie)

    @classmethod
    @dispatch
    def new(cls, selections: Dict):
        trie = Trie.new()
        for k, v in selections.items():
            assert isinstance(v, Selection)
            trie[k] = v
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


@dataclass
class Choice(Pytree):
    @abc.abstractmethod
    def filter(self, selection: Selection) -> "Choice":
        pass


@dataclass
class EmptyChoice(Choice):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return EmptyChoice()

    def filter(self, selection):
        return self

    def is_empty(self):
        return True

    @dispatch
    def merge(self, other):
        return other, self

    def __rich_tree__(self):
        return rich_tree.Tree("[bold](EmptyChoice)")


@dataclass
class ChoiceValue(Choice):
    value: Any

    def flatten(self):
        return (self.value,), ()

    @classmethod
    def new(cls, v):
        return ChoiceValue(v)

    def is_empty(self):
        return False

    def get_value(self):
        return self.value

    @dispatch
    def merge(self, other: "ChoiceValue"):
        return self, other

    @dispatch
    def filter(self, selection: AllSelection):
        return self

    @dispatch
    def filter(self, selection):
        return EmptyChoice()

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](ValueChoice)")
        tree.add(gpp.tree_pformat(self.value))
        return tree


@dataclass
class ChoiceMap(Choice):
    @abc.abstractmethod
    def get_submap(self, addr) -> Choice:
        pass

    @abc.abstractmethod
    def has_submap(self, addr) -> BoolArray:
        pass

    @abc.abstractmethod
    def is_empty(self) -> BoolArray:
        pass

    @abc.abstractmethod
    def merge(
        self,
        other: "ChoiceMap",
    ) -> Tuple["ChoiceMap", "ChoiceMap"]:
        pass

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
    ) -> "ChoiceMap":
        return EmptyChoice()

    @dispatch
    def filter(
        self,
        selection: Selection,
    ) -> "ChoiceMap":
        """Filter the addresses in a choice map, returning a new choice map.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen(genjax.Static)
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

    def safe_merge(self, other: "ChoiceMap") -> "ChoiceMap":
        new, discard = self.merge(other)
        if not discard.is_empty():
            raise Exception(f"Discard is non-empty.\n{discard}")
        return new

    def unsafe_merge(self, other: "ChoiceMap") -> "ChoiceMap":
        new, _ = self.merge(other)
        return new

    def get_choices(self):
        return self

    def strip(self):
        return strip(self)

    ###########
    # Dunders #
    ###########

    def __eq__(self, other):
        return self.flatten() == other.flatten()

    def __add__(self, other):
        return self.safe_merge(other)

    @dispatch
    def __getitem__(self, addrs: Tuple):
        submap = self.get_submap(addrs)
        if isinstance(submap, ChoiceValue):
            return submap.get_value()
        elif isinstance(submap, Mask):
            if isinstance(submap.value, ChoiceValue):
                return Mask(submap.mask, submap.value.get_value())
            return submap

    @dispatch
    def __getitem__(self, addr: Any):
        return self.__getitem__((addr,))


#########
# Trace #
#########


@dataclass
class Trace(Pytree):
    """> Abstract base class for traces of generative functions.

    A `Trace` is a data structure used to represent sampled executions
    of generative functions.

    Traces track metadata associated with log probabilities of choices,
    as well as other data associated with the invocation of a generative
    function, including the arguments it was invoked with, its return
    value, and the identity of the generative function itself.
    """

    @abc.abstractmethod
    def get_retval(self) -> Any:
        """Returns the return value from the generative function invocation
        which created the `Trace`.

        Examples:

            Here's an example using `genjax.normal` (a distribution). For distributions, the return value is the same as the (only) value in the returned choice map.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            retval = tr.get_retval()
            chm = tr.get_choices()
            v = chm.get_value()
            print(console.render((retval, v)))
            ```
        """

    @abc.abstractmethod
    def get_score(self) -> FloatArray:
        """Return the score of the `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen(genjax.Static)
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

    @abc.abstractmethod
    def get_args(self) -> Tuple:
        pass

    @abc.abstractmethod
    def get_choices(self) -> ChoiceMap:
        """Return a `ChoiceMap` representation of the set of traced random
        choices sampled during the execution of the generative function to
        produce the `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen(genjax.Static)
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x

            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            chm = tr.get_choices()
            print(console.render(chm))
            ```
        """

    @abc.abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        """Returns the generative function whose invocation created the
        `Trace`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

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
        """Given a `Selection`, return the total contribution to the score of
        the addresses contained within the `Selection`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen(genjax.Static)
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
        argdiffs = tree_diff_no_change(args)
        return gen_fn.update(key, self, choices, argdiffs)

    def get_aux(self) -> Tuple:
        raise NotImplementedError

    #################################
    # Default choice map interfaces #
    #################################

    def is_empty(self):
        return self.strip().is_empty()

    def filter(
        self,
        selection: Selection,
    ) -> Any:
        stripped = self.strip()
        filtered = stripped.filter(selection)
        return filtered

    def merge(self, other: ChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return self.strip().merge(other.strip())

    def has_submap(self, addr) -> BoolArray:
        choices = self.get_choices()
        return choices.has_submap(addr)

    def get_submap(self, addr) -> ChoiceMap:
        choices = self.get_choices()
        return choices.get_submap(addr)

    def get_selection(self):
        return self.strip().get_selection()

    def strip(self):
        """Remove all `Trace` metadata, and return a choice map.

        `ChoiceMap` instances produced by `tr.get_choices()` will preserve `Trace` instances. `strip` recursively calls `get_choices` to remove `Trace` instances.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            chm = tr.strip()
            print(console.render(chm))
            ```
        """
        return strip(self)

    def __getitem__(self, x):
        return self.get_choices()[x]


# Remove all trace metadata, and just return choices.
def strip(v):
    def _check(v):
        return isinstance(v, Trace)

    def _inner(v):
        if isinstance(v, Trace):
            return v.strip()
        else:
            return v

    return jtu.tree_map(_inner, v.get_choices(), is_leaf=_check)


###########
# Masking #
###########


@dataclass
class Mask(Pytree):
    """The `Mask` datatype provides access to the masking system. The masking
    system is heavily influenced by the functional `Option` monad.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as constraints in choice maps, and participate in inference computations (like scores, and importance weights or density ratios).

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    Users are expected to interact with `Mask` instances by either:

    * Unmasking them using the `Mask.unmask` interface. This interface uses JAX's `checkify` transformation to ensure that masked data exposed to a user is used only when valid. If a user chooses to `Mask.unmask` a `Mask` instance, they are also expected to use [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html) to transform their function to one which could return an error.

    * Using `Mask.match` - which allows a user to provide "none" and "some" lambdas. The "none" lambda should accept no arguments, while the "some" lambda should accept an argument whose type is the same as the masked value. These lambdas should return the same type (`Pytree`, array, etc) of value.
    """

    mask: BoolArray
    value: Any

    def flatten(self):
        return (self.mask, self.value), ()

    @classmethod
    def new(cls, mask: BoolArray, inner):
        if isinstance(inner, Mask):
            return Mask(
                jnp.logical_and(mask, inner.mask),
                inner.value,
            )
        else:
            return Mask(mask, inner)

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
            console = genjax.pretty()

            masked = genjax.mask(False, jnp.ones(5))
            v1 = masked.match(lambda: 10.0, lambda v: jnp.sum(v))
            masked = genjax.mask(True, jnp.ones(5))
            v2 = masked.match(lambda: 10.0, lambda v: jnp.sum(v))
            print(console.render((v1, v2)))
            ```
        """
        flag = jnp.array(self.mask)
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

        This operation is inherently unsafe with respect to inference semantics, and is only valid if the `Mask` is valid at runtime. To enforce validity checks, use `genjax.global_options.allow_checkify(True)` and then handle any code which utilizes `Mask.unmask` with [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html).

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import genjax
            console = genjax.pretty()

            masked = genjax.mask(True, jnp.ones(5))
            print(console.render(masked.unmask()))
            ```

            Here's an example which uses `jax.experimental.checkify`. To enable runtime checks, the user must enable them explicitly in `genjax`.

            ```python exec="yes" source="tabbed-left"
            import jax
            import jax.numpy as jnp
            import jax.experimental.checkify as checkify
            import genjax
            console = genjax.pretty()
            genjax.global_options.allow_checkify(True)

            masked = genjax.mask(False, jnp.ones(5))
            err, _ = checkify.checkify(masked.unmask)()
            print(console.render(err))

            genjax.global_options.allow_checkify(False)
            ```
        """

        # If a user chooses to `unmask`, require that they
        # jax.experimental.checkify.checkify their call in transformed
        # contexts.
        def _check():
            check_flag = jnp.all(self.mask)
            checkify.check(check_flag, "Mask is False, the masked value is invalid.\n")

        global_options.optional_check(_check)
        return self.value

    def unsafe_unmask(self):
        # Unsafe version of unmask -- should only be used internally.
        return self.value

    #########################
    # Choice map interfaces #
    #########################

    def is_empty(self):
        assert isinstance(self.value, ChoiceMap)
        return jnp.logical_and(self.mask, self.value.is_empty())

    def get_submap(self, addrs):
        assert isinstance(self.value, ChoiceMap)
        submap = self.value.get_submap(addrs)
        if isinstance(submap, EmptyChoice):
            return submap
        else:
            return Mask.new(self.mask, submap)

    def has_submap(self, addr):
        assert isinstance(self.value, ChoiceMap)
        check = self.value.has_submap(addr)
        return jnp.logical_and(self.mask, check)

    def filter(self, selection: Selection):
        choices = self.value.get_choices()
        assert isinstance(choices, Choice)
        return Mask.new(self.mask, choices.filter(selection))

    def get_choices(self):
        choices = self.value.get_choices()
        return Mask.new(self.mask, choices)

    ###########################
    # Address leaf interfaces #
    ###########################

    def get_value(self):
        assert isinstance(self.value, ChoiceValue)
        v = self.value.get_value()
        return Mask.new(self.mask, v)

    def try_value(self):
        if isinstance(self.value, ChoiceValue):
            return self.get_value()
        else:
            return self

    ###########
    # Dunders #
    ###########

    @dispatch
    def __eq__(self, other: "Mask"):
        return jnp.logical_and(
            jnp.logical_and(self.mask, other.mask),
            self.value == other.value,
        )

    @dispatch
    def __eq__(self, other: Any):
        return jnp.logical_and(
            self.mask,
            self.value == other,
        )

    @dispatch
    def __getitem__(self, addr: Any):
        masked = self.get_submap(addr)
        if isinstance(masked.value, ChoiceValue):
            return Mask(masked.mask, masked.value.get_value())
        else:
            return masked

    @dispatch
    def __getitem__(self, addrs: Tuple):
        masked = self.get_submap(addrs)
        if isinstance(masked.value, ChoiceValue):
            return Mask(masked.mask, masked.value.get_value())
        else:
            return masked

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        doc = gpp._pformat_array(self.mask, short_arrays=True)
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


@dataclass
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
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        """> Given a `key: PRNGKey` and arguments `x: Tuple`, the generative
        function sample a choice map $c \\sim p(\\cdot; x)$, as well as any
        untraced randomness $r \\sim p(\\cdot; x, c)$ to produce a trace $t =
        (x, c, r)$.

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
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            tr = genjax.normal.simulate(key, (0.0, 1.0))
            print(console.render(tr))
            ```

            Here's a slightly more complicated example using the `Static` generative function language. You can find more examples on the `Static` language page.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            @genjax.gen
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
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple[Choice, FloatArray, Any]:
        """> Given a `key: PRNGKey` and arguments ($x$), execute the generative
        function, returning a tuple containing the return value from the
        generative function call, the score ($s$) of the choice map assignment,
        and the choice map ($c$).

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
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            (chm, w, r) = genjax.normal.propose(key, (0.0, 1.0))
            print(console.render(chm))
            ```

            Here's a slightly more complicated example using the `Static` generative function language. You can find more examples on the `Static` language page.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import Static
            console = genjax.pretty()

            @gen(Static)
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
        chm = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        return (chm, score, retval)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        choice: Choice,
        args: Tuple,
    ) -> Tuple[Trace, FloatArray]:
        """> Given a `key: PRNGKey`, a choice map indicating constraints ($u$),
        and arguments ($x$), execute the generative function, and return an
        importance weight estimate of the conditional density evaluated at the
        non-constrained choices, and a trace whose choice map ($c = u' ⧺ u$) is
        consistent with the constraints ($u$), with unconstrained choices
        ($u'$) proposed from an internal proposal.

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
        self,
        key: PRNGKey,
        constraints: Mask,
        args: Tuple,
    ) -> Tuple[Trace, FloatArray]:
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
        self,
        key: PRNGKey,
        prev: Trace,
        new_constraints: Choice,
        diffs: Tuple,
    ) -> Tuple[Trace, FloatArray, Any, Choice]:
        raise NotImplementedError

    @dispatch
    def update(
        self,
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
            if possible_discards.is_empty():
                discard = EmptyChoice()
            else:
                # We return the possible_discards, but denote them as invalid via masking.
                discard = mask(False, possible_discards)
            primal = tree_diff_primal(retdiff)
            retdiff = tree_diff_unknown_change(primal)
            return (new_tr, w, retdiff, discard)

        def _some(chm):
            (new_tr, w, retdiff, _) = self.update(key, prev, chm, argdiffs)
            if possible_discards.is_empty():
                discard = EmptyChoice()
            else:
                # The true_discards should match the Pytree type of possible_discards,
                # but these are valid.
                discard = mask(True, possible_discards)
            primal = tree_diff_primal(retdiff)
            retdiff = tree_diff_unknown_change(primal)
            return (new_tr, w, retdiff, discard)

        return new_constraints.match(_none, _some)

    def assess(
        self,
        choice: Choice,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        """> Given a complete choice map indicating constraints ($u$) for all
        choices, and arguments ($x$), execute the generative function, and
        return the return value of the invocation, and the score of the choice
        map ($s$).

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


@dataclass
class JAXGenerativeFunction(GenerativeFunction, Pytree):
    """A `GenerativeFunction` subclass for JAX compatible generative
    functions."""

    # This is used to support tracing.
    # Below, a default implementation: GenerativeFunctions
    # may customize this to improve compilation time.
    def __abstract_call__(self, *args) -> Any:
        # This should occur only during abstract evaluation,
        # the fact that the value has type PRNGKey is all that matters.
        key = jax.random.PRNGKey(0)
        tr = self.simulate(key, args)
        retval = tr.get_retval()
        return retval

    @typecheck
    def unzip(
        self,
        fixed: Choice,
    ) -> Tuple[Callable[[Choice, Tuple], FloatArray], Callable[[Choice, Tuple], Any],]:
        def score(differentiable: Tuple, nondifferentiable: Tuple) -> FloatArray:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            (score, _) = self.assess(merged, args)
            return score

        def retval(differentiable: Tuple, nondifferentiable: Tuple) -> Any:
            provided, args = tree_zipper(differentiable, nondifferentiable)
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
        grad, nograd = tree_grad_split(
            (chm, trace.get_args()),
        )
        choice_gradient_tree, _ = jax.grad(scorer)(grad, nograd)
        return choice_gradient_tree


########################
# Concrete choice maps #
########################


@dataclass
class HierarchicalChoiceMap(ChoiceMap):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @classmethod
    @dispatch
    def new(cls, constraints: Dict):
        assert isinstance(constraints, Dict)
        trie = Trie.new()
        for k, v in constraints.items():
            v = (
                ChoiceValue(v)
                if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
                else v
            )
            trie[k] = v
        return HierarchicalChoiceMap(trie)

    @classmethod
    @dispatch
    def new(cls, trie: Trie):
        check = trie.is_empty()
        if static_check_is_concrete(check) and check:
            return EmptyChoice()
        else:
            return HierarchicalChoiceMap(trie)

    @classmethod
    @dispatch
    def new(cls):
        trie = Trie.new()
        return HierarchicalChoiceMap(trie)

    def is_empty(self):
        return self.trie.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        def _inner(k, v):
            sub = selection.get_subselection(k)
            under = v.filter(sub)
            return k, under

        trie = Trie.new()
        iter = self.get_submaps_shallow()
        for k, v in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoice):
                trie[k] = v

        new = HierarchicalChoiceMap(trie)
        if new.is_empty():
            return EmptyChoice()
        else:
            return new

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
        trie = Trie.new()
        for k, v in self.get_submaps_shallow():
            trie[k] = v.get_selection()
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

    ###########
    # Dunders #
    ###########

    def __setitem__(self, k, v):
        v = (
            ChoiceValue(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.trie[k] = v

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


@dataclass
class DisjointUnionChoiceMap(ChoiceMap):
    """> A choice map combinator type which represents a disjoint union over
    multiple choice maps.

    The internal data representation of a `ChoiceMap` is often specialized to support optimized code generation for inference interfaces, but the address hierarchy which a `ChoiceMap` represents (as an assignment of choices to addresses) must be generic.

    To make this more concrete, a `VectorChoiceMap` represents choices with addresses of the form `(integer_index, ...)` - but its internal data representation is a struct-of-arrays. A `HierarchicalChoiceMap` can also represent address assignments with form `(integer_index, ...)` - but supporting choice map interfaces like `merge` across choice map types with specialized internal representations is complicated.

    Modeling languages might also make use of specialized representations for (JAX compatible) address uncertainty -- and addresses can contain runtime data e.g. `Static` generative functions can support addresses `(dynamic_integer_index, ...)` where the index is not known at tracing time. When generative functions mix `(static_integer_index, ...)` and `(dynamic_integer_index, ...)` - resulting choice maps must be a type of disjoint union, whose methods include branching decisions on runtime data.

    To this end, `DisjointUnionChoiceMap` is a `ChoiceMap` type designed to support disjoint unions of choice maps of different types. It supports implementations of the choice map interfaces which are generic over the type of choice maps in the union, and also works with choice maps that contain runtime resolved address data.
    """

    submaps: List[ChoiceMap]

    def flatten(self):
        return (self.submaps,), ()

    @classmethod
    def new(cls, submaps: List[ChoiceMap]):
        if not submaps:
            return EmptyChoice()
        else:
            return DisjointUnionChoiceMap(submaps)

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
            return DisjointUnionChoiceMap.new(new_submaps)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](DisjointUnionChoiceMap)")
        for submap in self.submaps:
            sub_tree = submap.__rich_tree__()
            tree.add(sub_tree)
        return tree


########################
# Language constructor #
########################


@dataclass
class LanguageConstructor(Pytree):
    """A `LanguageConstructor` is a type which can be used to construct
    generative function instances."""

    constructor: Callable

    def flatten(self):
        return (), (self.constructor,)

    def __call__(self, *args, **kwargs):
        return self.constructor(*args, **kwargs)


##############
# Shorthands #
##############

# Choices and choice maps
empty_choice = EmptyChoice.new
choice_value = ChoiceValue.new
hierarchical_choice_map = HierarchicalChoiceMap.new
disjoint_union_choice_map = DisjointUnionChoiceMap.new

# Selections
none_select = NoneSelection.new
all_select = AllSelection.new
hierarchical_select = HierarchicalSelection.new


# Functional types
mask = Mask.new


@dispatch
def choice_map():
    return hierarchical_choice_map()


@dispatch
def choice_map(v: Any):
    return choice_value(v)


@dispatch
def choice_map(submaps: Dict):
    return hierarchical_choice_map(submaps)


@dispatch
def choice_map(submaps: List[ChoiceMap]):
    return disjoint_union_choice_map(submaps)


# TODO: experimental for dynamic addresses.
# @dispatch
# def choice_map(addrs: List[Any], submaps: List[ChoiceMap]):
#    return dynamic_choice_map(addrs, submaps)


@dispatch
def select():
    return hierarchical_select()


@dispatch
def select(subselects: Dict):
    return hierarchical_select(subselects)


@dispatch
def select(*addrs: Any):
    return hierarchical_select(*addrs)
