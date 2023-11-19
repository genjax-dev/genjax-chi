# Copyright 2022 MIT Probabilistic Computing Project
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
import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import rich
from jax.experimental.checkify import checkify

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.hashable_dict import hashable_dict
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.incremental import tree_diff_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.interpreters.incremental import tree_diff_unknown_change
from genjax._src.core.pytree.checks import (
    static_check_tree_leaves_have_matching_leading_dim,
)
from genjax._src.core.pytree.checks import static_check_tree_structure_equivalence
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.string import PytreeString
from genjax._src.core.pytree.utilities import tree_grad_split
from genjax._src.core.pytree.utilities import tree_zipper
from genjax._src.core.typing import Any
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import String
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import static_check_is_concrete
from genjax._src.core.typing import typecheck
from genjax._src.global_options import global_options


########################
# Generative datatypes #
########################


#############
# Selection #
#############


@dataclasses.dataclass
class Selection(Pytree):
    @abc.abstractmethod
    def complement(self) -> "Selection":
        """Return a `Selection` which filters addresses to the complement set
        of the provided `Selection`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli
            console = genjax.pretty()

            @genjax.gen
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

    def get_selection(self):
        return self

    ###########
    # Dunders #
    ###########

    def __getitem__(self, addr):
        submap = self.get_submap(addr)
        return submap

    ###################
    # Pretty printing #
    ###################

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = rich.tree.Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


############################
# Concrete leaf selections #
############################


@dataclasses.dataclass
class NoneSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return NoneSelection()

    def complement(self):
        return AllSelection()

    def get_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf value."
        )

    def set_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf choice value."
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        tree = tree.add("[bold](None)")
        return tree


@dataclasses.dataclass
class AllSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return AllSelection()

    def complement(self):
        return NoneSelection()

    def get_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf value."
        )

    def set_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf choice value."
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        tree = tree.add("[bold](All)")
        return tree


##################################
# Concrete structured selections #
##################################


@dataclasses.dataclass
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

    def complement(self):
        return ComplementHierarchicalSelection(self.trie)

    def has_submap(self, addr):
        return self.trie.has_submap(addr)

    def get_submap(self, addr):
        value = self.trie.get_submap(addr)
        if value is None:
            return NoneSelection()
        else:
            subselect = value.get_selection()
            if isinstance(subselect, Trie):
                return HierarchicalSelection(subselect)
            else:
                return subselect

    def get_submaps_shallow(self):
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

    def __rich_tree__(self, tree):
        for k, v in self.get_submaps_shallow():
            subk = tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        return tree


@dataclasses.dataclass
class ComplementHierarchicalSelection(HierarchicalSelection):
    trie: Trie

    def flatten(self):
        return (self.selection,), ()

    def complement(self):
        return HierarchicalSelection(self.trie)

    def has_submap(self, addr):
        return jnp.logical_not(self.trie.has_submap(addr))

    def get_submap(self, addr):
        value = self.trie.get_submap(addr)
        if value is None:
            return AllSelection()
        else:
            subselect = value.get_selection()
            if isinstance(subselect, Trie):
                return HierarchicalSelection(subselect).complement()
            else:
                return subselect.complement()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Complement)")
        for k, v in self.get_submaps_shallow():
            subk = sub_tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        tree.add(sub_tree)
        return tree


@dataclass
class IndexedSelection(Selection):
    indices: IntArray
    inner: Selection

    def flatten(self):
        return (
            self.indices,
            self.inner,
        ), ()

    @classmethod
    @dispatch
    def new(cls, idx: Union[Int, IntArray]):
        idxs = jnp.array(idx)
        return IndexedSelection(idxs, AllSelection())

    @classmethod
    @dispatch
    def new(cls, idx: Any, inner: Selection):
        idxs = jnp.array(idx)
        return IndexedSelection(idxs, inner)

    @classmethod
    @dispatch
    def new(cls, idx: Any, *inner: Any):
        idxs = jnp.array(idx)
        inner = select(*inner)
        return IndexedSelection(idxs, inner)

    @dispatch
    def has_submap(self, addr: IntArray):
        return jnp.isin(addr, self.indices)

    @dispatch
    def has_submap(self, addr: Tuple):
        if len(addr) <= 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_submap(addr))

    def get_submap(self, addr):
        return self.index_selection.get_submap(addr)

    def complement(self):
        return ComplementIndexedSelection(self)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.indices, short_arrays=True)
        sub_tree = rich.tree.Tree(f"[bold](Indexed,{doc})")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


@dataclass
class ComplementIndexedSelection(IndexedSelection):
    index_selection: Selection

    def __init__(self, index_selection):
        self.index_selection = index_selection

    def flatten(self):
        return (self.index_selection,), ()

    @dispatch
    def has_submap(self, addr: IntArray):
        return jnp.logical_not(jnp.isin(addr, self.indices))

    @dispatch
    def has_submap(self, addr: Tuple):
        if len(addr) <= 1:
            return False
        (idx, addr) = addr
        return jnp.logical_not(
            jnp.logical_and(idx in self.indices, self.inner.has_submap(addr))
        )

    def get_submap(self, addr):
        return self.index_selection.get_submap(addr).complement()

    def complement(self):
        return self.index_selection

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Complement)")
        self.index_selection.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


###########
# Choices #
###########


@dataclasses.dataclass
class Choice(Pytree):
    @abc.abstractmethod
    def filter(self, selection: Selection) -> "Choice":
        pass


@dataclasses.dataclass
class EmptyChoice(Choice):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return EmptyChoice()

    def filter(self, selection):
        return self

    def __rich_tree__(self, tree):
        sub = rich.tree.Tree("[bold](Empty)")
        tree.add(sub)
        return tree


@dataclasses.dataclass
class ChoiceValue(Choice):
    value: Any

    def flatten(self):
        return (self.value,), ()

    @classmethod
    def new(cls, v):
        return ChoiceValue(v)

    def get_value(self):
        return self.value

    @dispatch
    def filter(self, selection: AllSelection):
        return self

    @dispatch
    def filter(self, selection: NoneSelection):
        return EmptyChoice()

    def __rich_tree__(self, tree):
        sub = rich.tree.Tree("[bold](Value)")
        sub.add(self.value)
        tree.add(sub)
        return tree


@dataclasses.dataclass
class ChoiceMap(Choice):
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

            @genjax.gen
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

    ###########
    # Dunders #
    ###########

    def __eq__(self, other):
        return self.flatten() == other.flatten()

    # Optional: mutable setter.
    def __setitem__(self, key, value):
        raise Exception(
            f"ChoiceMap of type {type(self)} does not implement __setitem__.",
        )

    def __add__(self, other):
        return self.safe_merge(other)

    def __getitem__(self, addr):
        submap = self.get_submap(addr)
        if isinstance(submap, ChoiceValue):
            return submap.get_value()

    ###################
    # Pretty printing #
    ###################

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = rich.tree.Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


#########
# Trace #
#########


@dataclasses.dataclass
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

            @genjax.gen
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

            @genjax.gen
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

            @genjax.gen
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
        choices: ChoiceMap,
        argdiffs: Tuple,
    ):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, choices, argdiffs)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        choices: ChoiceMap,
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

        def _check(v):
            return isinstance(v, Trace)

        def _inner(v):
            if isinstance(v, Trace):
                return v.strip()
            else:
                return v

        return jtu.tree_map(_inner, self.get_choices(), is_leaf=_check)


###########
# Masking #
###########


class Mask(Pytree):
    """The `Mask` datatype provides access to the masking system. The masking
    system is heavily influenced by the functional `Option` monad.

    Masks can be used in a variety of ways as part of generative computations - their primary role is to denote data which is valid under inference computations. Valid data can be used as constraints in choice maps, and participate in inference computations (like scores, and importance weights or density ratios).

    Masks are also used internally by generative function combinators which include uncertainty over structure.

    Users are expected to interact with `Mask` instances by either:

    * Unmasking them using the `Mask.unmask` interface. This interface uses JAX's `checkify` transformation to ensure that masked data exposed to a user is used only when valid. If a user chooses to `Mask.unmask` a `Mask` instance, they are also expected to use [`jax.experimental.checkify.checkify`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.checkify.checkify.html) to transform their function to one which could return an error.

    * Using `Mask.match` - which allows a user to provide "none" and "some" lambdas. The "none" lambda should accept no arguments, while the "some" lambda should accept an argument whose type is the same as the masked value. These lambdas should return the same type (`Pytree`, array, etc) of value.
    """

    def __init__(self, mask: BoolArray, value: Any):
        self.mask = mask
        self.value = value

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

    def get_submap(self, addr):
        assert isinstance(self.value, ChoiceMap)
        submap = self.value.get_submap(addr)
        if isinstance(submap, EmptyChoice):
            return submap
        else:
            return Mask.new(self.mask, submap)

    def has_submap(self, addr):
        assert isinstance(self.value, ChoiceMap)
        check = self.value.has_submap(addr)
        return jnp.logical_and(self.mask, check)

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

    def __hash__(self):
        hash1 = hash(self.value)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.mask, short_arrays=True)
        sub_tree = rich.tree.Tree(f"[bold](Mask, {doc})")
        if isinstance(self.value, Pytree):
            _ = self.value.__rich_tree__(sub_tree)
        else:
            val_tree = gpp.tree_pformat(self.value, short_arrays=True)
            sub_tree.add(val_tree)
        tree.add(sub_tree)
        return tree

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = rich.tree.Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


#################
# Tagged unions #
#################


@dataclass
class TaggedUnion(Pytree):
    tag: IntArray
    values: List[Any]

    def flatten(self):
        return (self.tag, self.values), ()

    @classmethod
    @typecheck
    def new(cls, tag: IntArray, values: List[Any]):
        return cls(tag, values)

    def _static_assert_tagged_union_switch_num_callables_is_num_values(self, callables):
        assert len(callables) == len(self.values)

    def _static_assert_tagged_union_switch_returns_same_type(self, vs):
        return True

    @typecheck
    def match(self, *callables: Callable):
        assert len(callables) == len(self.values)
        self._static_assert_tagged_union_switch_num_callables_is_num_values(callables)
        vs = list(map(lambda v: v[0](v[1]), zip(callables, self.values)))
        self._static_assert_tagged_union_switch_returns_same_type(vs)
        vs = jnp.array(vs)
        return vs[self.tag]

    ###########
    # Dunders #
    ###########

    def __getattr__(self, name):
        subs = list(map(lambda v: getattr(v, name), self.values))
        if subs and all(map(lambda v: isinstance(v, Callable), subs)):

            def wrapper(*args):
                vs = [s(*args) for s in subs]
                return TaggedUnion(self.tag, vs)

            return wrapper
        else:
            return TaggedUnion(self.tag, subs)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.tag, short_arrays=True)
        vals_tree = gpp.tree_pformat(self.values, short_arrays=True)
        sub_tree = rich.tree.Tree(f"[bold](TaggedUnion, {doc})")
        sub_tree.add(vals_tree)
        tree.add(sub_tree)
        return tree

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = rich.tree.Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


#####
# Generative function
#####


@dataclasses.dataclass
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
    ) -> Tuple[Any, FloatArray, ChoiceMap]:
        """> Given a `key: PRNGKey` and arguments ($x$), execute the generative
        function, returning a tuple containing the return value from the
        generative function call, the score ($s$) of the choice map assignment,
        and the choice map ($c$).

        The default implementation just calls `simulate`, and then extracts the data from the `Trace` returned by `simulate`. Custom generative functions can overload the implementation for their own uses (e.g. if they don't have an associated `Trace` datatype, but can be used as a proposal).

        Arguments:
            key: A `PRNGKey`.
            args: Arguments to the generative function.

        Returns:
            retval: the return value from the generative function invocation
            s: the score ($s$) of the choice map assignment
            chm: the choice map assignment ($c$)

        Examples:

            Here's an example using a `genjax` distribution (`normal`). Distributions are generative functions, so they support the interface.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            (r, w, chm) = genjax.normal.propose(key, (0.0, 1.0))
            print(console.render(chm))
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
            (r, w, chm) = model.propose(key, ())
            print(console.render(chm))
            ```
        """
        tr = self.simulate(key, args)
        chm = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        return (retval, score, chm)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Trace]:
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
            w: An importance weight.
            tr: A trace capturing the data and inference data associated with the generative function invocation.

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
    ) -> Tuple[FloatArray, Trace]:
        def _inactive():
            w = 0.0
            tr = self.simulate(key, args)
            return w, tr

        def _active(chm):
            w, tr = self.importance(key, chm, args)
            return w, tr

        return constraints.match(_inactive, _active)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        new_constraints: ChoiceMap,
        diffs: Tuple,
    ) -> Tuple[Any, FloatArray, Trace, ChoiceMap]:
        raise NotImplementedError

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        new_constraints: Mask,
        argdiffs: Tuple,
    ) -> Tuple[Any, FloatArray, Trace, Mask]:
        # The semantics of the merge operation entail that the second returned value
        # is the discarded values after the merge.
        discard_option = prev.strip()
        possible_constraints = new_constraints.unsafe_unmask()
        _, possible_discards = discard_option.merge(possible_constraints)

        def _none():
            (retdiff, w, new_tr, _) = self.update(key, prev, EmptyChoice(), argdiffs)
            if possible_discards.is_empty():
                discard = EmptyChoice()
            else:
                # We return the possible_discards, but denote them as invalid via masking.
                discard = mask(False, possible_discards)
            primal = tree_diff_primal(retdiff)
            retdiff = tree_diff_unknown_change(primal)
            return (retdiff, w, new_tr, discard)

        def _some(chm):
            (retdiff, w, new_tr, _) = self.update(key, prev, chm, argdiffs)
            if possible_discards.is_empty():
                discard = EmptyChoice()
            else:
                # The true_discards should match the Pytree type of possible_discards,
                # but these are valid.
                discard = mask(True, possible_discards)
            primal = tree_diff_primal(retdiff)
            retdiff = tree_diff_unknown_change(primal)
            return (retdiff, w, new_tr, discard)

        return new_constraints.match(_none, _some)

    def assess(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Any, FloatArray]:
        """> Given a `key: PRNGKey`, a complete choice map indicating
        constraints ($u$) for all choices, and arguments ($x$), execute the
        generative function, and return the return value of the invocation, and
        the score of the choice map ($s$).

        Arguments:
            key: A `PRNGKey`.
            chm: A complete choice map indicating constraints ($u$) for all choices.
            args: Arguments to the generative function ($x$).

        Returns:
            retval: The return value from the generative function invocation.
            score: The score of the choice map.

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


@dataclasses.dataclass
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

    def unzip(
        self,
        key: PRNGKey,
        fixed: ChoiceMap,
    ) -> Tuple[
        Callable[[ChoiceMap, Tuple], FloatArray],
        Callable[[ChoiceMap, Tuple], Any],
    ]:
        def score(differentiable: Tuple, nondifferentiable: Tuple) -> FloatArray:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            (_, score) = self.assess(key, merged, args)
            return score

        def retval(differentiable: Tuple, nondifferentiable: Tuple) -> Any:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            (retval, _) = self.assess(key, merged, args)
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


@dataclasses.dataclass
class DynamicHierarchicalChoiceMap(ChoiceMap):
    dynamic_addrs: List[Any]
    submaps: List[ChoiceMap]

    def flatten(self):
        return (self.dynamic_addrs, self.submaps), ()

    @classmethod
    @dispatch
    def new(cls, dynamic_addrs, submaps):
        return DynamicHierarchicalChoiceMap(dynamic_addrs, submaps)

    @classmethod
    @dispatch
    def new(cls):
        return DynamicHierarchicalChoiceMap([], [])

    def is_empty(self):
        return jnp.all(jnp.array(map(lambda v: v.is_empty(), self.submaps)))

    @dispatch
    def get_submap(self, addr: IntArray):
        compares = map(lambda v: addr == v, self.dynamic_addrs)
        masks = list(map(lambda v: mask(v[0], v[1]), zip(compares, self.submaps)))
        return DisjointUnionChoiceMap(masks)

    @dispatch
    def get_submap(self, addr: Tuple):
        (idx, rest) = addr
        disjoint_union_chm = self.get_submap(idx)
        return disjoint_union_chm.get_submap(rest)

    @dispatch
    def has_submap(self, addr: IntArray):
        equality_checks = jnp.array(map(lambda v: addr == v, self.dynamic_addrs))
        return jnp.any(equality_checks)

    @dispatch
    def has_submap(self, addr: Tuple):
        (idx, rest) = addr
        disjoint_union_chm = self.get_submap(idx)
        return jnp.logical_and(
            self.has_submap(idx),
            disjoint_union_chm.has_submap(rest),
        )

    ###########
    # Dunders #
    ###########

    @dispatch
    def __setitem__(self, k: Any, v: Any):
        v = (
            ChoiceValue(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.dynamic_addrs.append(k)
        self.submaps.append(v)

    @dispatch
    def __setitem__(self, k: String, v: Any):
        v = (
            ChoiceValue(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.dynamic_addrs.append(PytreeString(k))
        self.submaps.append(v)

    @dispatch
    def __setitem__(self, k: Tuple, v: Any):
        v = (
            ChoiceValue(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.dynamic_addrs.append(k)
        self.submaps.append(v)


def tree_choice_map_specialize(v):
    def _convert(v):
        if isinstance(v, DynamicHierarchicalChoiceMap):
            return v.maybe_specialize()
        else:
            return v

    return jtu.tree_map(
        _convert, v, is_leaf=lambda v: isinstance(v, DynamicHierarchicalChoiceMap)
    )


class DynamicConvertible:
    @abc.abstractmethod
    def dynamic_convert(self) -> DynamicHierarchicalChoiceMap:
        pass


@dataclasses.dataclass
class HierarchicalChoiceMap(ChoiceMap, DynamicConvertible):
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
            sub = selection.get_submap(k)
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

    @dispatch
    def replace(
        self,
        selection: HierarchicalSelection,
        replacement: ChoiceMap,
    ) -> ChoiceMap:
        complement = self.filter(selection.complement())
        return complement.unsafe_merge(replacement)

    @dispatch
    def insert(
        self,
        selection: HierarchicalSelection,
        extension: ChoiceMap,
    ) -> ChoiceMap:
        raise NotImplementedError

    def has_submap(self, addr):
        return self.trie.has_submap(addr)

    def _lift_value(self, value):
        if value is None:
            return EmptyChoice()
        else:
            submap = value.get_choices()
            if isinstance(submap, Trie):
                return HierarchicalChoiceMap(submap)
            else:
                return submap

    @dispatch
    def get_submap(self, addr: Any):
        value = self.trie.get_submap(addr)
        return self._lift_value(value)

    @dispatch
    def get_submap(self, addr: IntArray):
        if static_check_is_concrete(addr):
            value = self.trie.get_submap(addr)
            return self._lift_value(value)
        else:
            dynamic_chm = self.dynamic_convert()
            return dynamic_chm.get_submap(addr)

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
            submap = v[1].get_choices()
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

    def dynamic_convert(self) -> DynamicHierarchicalChoiceMap:
        dynamic_addrs = []
        submaps = []
        for k, v in self.get_submaps_shallow():
            dynamic_addrs.append(k)
            submaps.append(v)
        return DynamicHierarchicalChoiceMap(dynamic_addrs, submaps)

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

    def __hash__(self):
        return hash(self.trie)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Hierarchical)")
        for k, v in self.get_submaps_shallow():
            subk = sub_tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        tree.add(sub_tree)
        return tree


@dataclass
class IndexedChoiceMap(ChoiceMap):
    indices: IntArray
    inner: ChoiceMap

    def flatten(self):
        return (self.indices, self.inner), ()

    @classmethod
    @dispatch
    def new(cls, indices: IntArray, inner: ChoiceMap) -> ChoiceMap:
        # Promote raw integers (or scalars) to non-null leading dim.
        indices = jnp.array(indices, copy=False)

        # Verify that dimensions are consistent before creating an
        # `IndexedChoiceMap`.
        _ = static_check_tree_leaves_have_matching_leading_dim((inner, indices))

        # if you try to wrap around an EmptyChoice, do nothing.
        if isinstance(inner, EmptyChoice):
            return inner

        return IndexedChoiceMap(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: List, inner: ChoiceMap) -> ChoiceMap:
        indices = jnp.array(indices)
        return IndexedChoiceMap.new(indices, inner)

    @classmethod
    @dispatch
    def new(cls, indices: Any, inner: Dict) -> ChoiceMap:
        inner = choice_map(inner)
        return IndexedChoiceMap.new(indices, inner)

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        return IndexedChoiceMap(self.indices, self.inner.filter(selection))

    def has_submap(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_submap(addr))

    @dispatch
    def filter(
        self,
        selection: IndexedSelection,
    ) -> ChoiceMap:
        flags = jnp.isin(selection.indices, self.indices)
        filtered_inner = self.inner.filter(selection.inner)
        masked = mask(flags, filtered_inner)
        return IndexedChoiceMap(self.indices, masked)

    def has_submap(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_submap(addr))

    @dispatch
    def get_submap(self, addr: Tuple):
        if len(addr) == 1:
            return self.get_submap(addr[0])
        idx, *rest = addr
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        submap = submap.get_submap(tuple(rest))
        if isinstance(submap, EmptyChoice):
            return submap
        else:
            return mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, idx: IntArray):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, idx: Int):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        submap = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return mask(jnp.isin(idx, self.indices), submap)

    @dispatch
    def get_submap(self, addr: Any):
        return EmptyChoice()

    def get_selection(self):
        return self.inner.get_selection()

    def merge(self, _: ChoiceMap):
        raise Exception("TODO: can't merge IndexedChoiceMaps")

    def get_index(self):
        return self.indices

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.indices, short_arrays=True)
        sub_tree = rich.tree.Tree(f"[bold](Indexed,{doc})")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


@dataclasses.dataclass
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
        return DisjointUnionChoiceMap(submaps)

    def has_submap(self, addr):
        checks = jnp.array(map(lambda v: v.has_submap(addr), self.submaps))
        return jnp.sum(checks) == 1

    def get_submap(self, addr):
        # Tracer time computation which eliminates the DisjointUnionChoiceMap
        # type, and returns a more specialized choice map.
        new_submaps = list(
            filter(
                lambda v: not isinstance(v, EmptyChoice),
                map(lambda v: v.get_submap(addr), self.submaps),
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

        # We've reached the `ChoiceValue` level - now we need to perform checks
        # * There must at least one valid leaf instance.
        # * There should be only one valid leaf instance.
        elif all(check_address_leaves):
            # Convert all to Mask instances -- this operation will preserve existing
            # Mask instances (and jnp.logical_and with the provided flag).
            new_submaps = list(map(lambda v: mask(True, v), new_submaps))

            # Now, extract the flags and runtime check that at least one is valid.
            masks = jnp.array(
                list(map(lambda v: v.mask, new_submaps)),
            )

            def _check_valid():
                check_flag = jnp.any(masks)
                return checkify.check(
                    check_flag,
                    "(DisjointUnionChoiceMap.get_submap): masked leaf values have no valid data.",
                )

            global_options.optional_check(_check_valid)

            # Now, extract the flags and runtime check that only one is valid.
            def _check_only_one():
                check_flag = jnp.sum(masks) == 1
                return checkify.check(
                    check_flag,
                    "(DisjointUnionChoiceMap.get_submap): multiple valid leaves.",
                )

            global_options.optional_check(_check_only_one)

            # Get the index of the valid value, and create a `TaggedUnion` with it.
            tag = jnp.argwhere(masks, size=1).reshape(1)[0]
            new_submaps = list(map(lambda v: v.unsafe_unmask(), new_submaps))
            return tagged_union(tag, new_submaps)
        else:
            return DisjointUnionChoiceMap(new_submaps)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](DisjointUnion)")
        for submap in self.submaps:
            _ = submap.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


@dataclasses.dataclass
class DisjointPairChoiceMap(ChoiceMap):
    """> A choice map combinator type which represents a disjoint union of a
    pair of choice maps.

    The internal data representation of a `ChoiceMap` is often specialized to support optimized code generation for inference interfaces, but the address hierarchy which a `ChoiceMap` represents (as an assignment of choices to addresses) must be generic.

    To make this more concrete, a `VectorChoiceMap` represents choices with addresses of the form `(integer_index, ...)` - but its internal data representation is a struct-of-arrays. A `HierarchicalChoiceMap` can also represent address assignments with form `(integer_index, ...)` - but supporting choice map interfaces like `merge` across choice map types with specialized internal representations is complicated.

    Modeling languages might also make use of specialized representations for (JAX compatible) address uncertainty -- and addresses can contain runtime data e.g. `Static` generative functions can support addresses `(dynamic_integer_index, ...)` where the index is not known at tracing time. When generative functions mix `(static_integer_index, ...)` and `(dynamic_integer_index, ...)` - resulting choice maps must be a type of disjoint union, whose methods include branching decisions on runtime data.

    To this end, `DisjointPairChoiceMap` is a `ChoiceMap` type designed to support disjoint unions of choice maps of different types. It supports implementations of the choice map interfaces which are generic over the type of choice maps in the union, and also works with choice maps that contain runtime resolved address data.
    """

    submap_1: ChoiceMap
    submap_2: ChoiceMap

    def flatten(self):
        return (self.submap_1, self.submap_2), ()

    @classmethod
    def new(cls, submap_1: ChoiceMap, submap_2: ChoiceMap):
        return DisjointPairChoiceMap(submap_1, submap_2)

    def has_submap(self, addr):
        checks = jnp.array(
            map(lambda v: v.has_submap(addr), [self.submap_1, self.submap_2])
        )
        return jnp.sum(checks) == 1

    def get_submap(self, addr):
        # Tracer time computation which eliminates the DisjointPairChoiceMap
        # type, and returns a more specialized choice map.
        new_submaps = list(
            filter(
                lambda v: not isinstance(v, EmptyChoice),
                map(lambda v: v.get_submap(addr), self.submaps),
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

        # We've reached the `ChoiceValue` level - now we need to perform checks
        # * There must at least one valid leaf instance.
        # * There should be only one valid leaf instance.
        elif all(check_address_leaves):
            # Convert all to Mask instances -- this operation will preserve existing
            # Mask instances (and jnp.logical_and with the provided flag).
            new_submaps = list(map(lambda v: mask(True, v), new_submaps))

            # Now, extract the flags and runtime check that at least one is valid.
            masks = jnp.array(
                list(map(lambda v: v.mask, new_submaps)),
            )

            def _check_valid():
                check_flag = jnp.any(masks)
                return checkify.check(
                    check_flag,
                    "(DisjointUnionChoiceMap.get_submap): masked leaf values have no valid data.",
                )

            global_options.optional_check(_check_valid)

            # Now, extract the flags and runtime check that only one is valid.
            def _check_only_one():
                check_flag = jnp.sum(masks) == 1
                return checkify.check(
                    check_flag,
                    "(DisjointUnionChoiceMap.get_submap): multiple valid leaves.",
                )

            global_options.optional_check(_check_only_one)

            # Get the index of the valid value, and create a `TaggedUnion` with it.
            tag = jnp.argwhere(masks, size=1).reshape(1)[0]
            new_submaps = list(map(lambda v: v.unsafe_unmask(), new_submaps))
            return tagged_union(tag, new_submaps)
        else:
            return DisjointUnionChoiceMap(new_submaps)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](DisjointPair)")
        _ = self.submap_1.__rich_tree__(sub_tree)
        _ = self.submap_2.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


##############
# Shorthands #
##############

empty_choice_map = EmptyChoice.new
value_choice_map = ChoiceValue.new
all_select = AllSelection.new
none_select = NoneSelection.new
hierarchical_choice_map = HierarchicalChoiceMap.new
hierarchical_select = HierarchicalSelection.new
indexed_choice_map = IndexedChoiceMap.new
indexed_select = IndexedSelection.new
disjoint_union_choice_map = DisjointUnionChoiceMap.new
disjoint_pair_choice_map = DisjointPairChoiceMap.new
dynamic_choice_map = DynamicHierarchicalChoiceMap.new
mask = Mask.new
tagged_union = TaggedUnion.new


@dispatch
def choice_map():
    return hierarchical_choice_map()


@dispatch
def choice_map(v: Any):
    return value_choice_map(v)


@dispatch
def choice_map(submaps: Dict):
    return hierarchical_choice_map(submaps)


@dispatch
def choice_map(indices: List[Int], submaps: List[ChoiceMap]):
    # submaps must have same Pytree structure to use
    # optimized representation.
    if static_check_tree_structure_equivalence(submaps):
        index_arr = jnp.array(indices)
        return indexed_choice_map(index_arr, submaps)
    else:
        return dynamic_choice_map(indices, submaps)


@dispatch
def choice_map(index: Int, submap: ChoiceMap):
    expanded = jtu.tree_map(lambda v: jnp.expand_dims(v, axis=0), submap)
    return indexed_choice_map([index], expanded)


@dispatch
def choice_map(submaps: List[ChoiceMap]):
    return disjoint_union_choice_map(submaps)


@dispatch
def choice_map(submap_1: ChoiceMap, submap_2: ChoiceMap):
    return disjoint_pair_choice_map(submap_1, submap_2)


@dispatch
def choice_map(dynamic_addrs: List[Any], submaps: List[ChoiceMap]):
    return dynamic_choice_map(dynamic_addrs, submaps)


@dispatch
def select():
    return all_select()


@dispatch
def select(subselects: Dict):
    return hierarchical_select(subselects)


@dispatch
def select(*addrs: Any):
    return hierarchical_select(*addrs)


@dispatch
def select(idx: Union[Int, IntArray], *args):
    return indexed_select(idx, *args)
