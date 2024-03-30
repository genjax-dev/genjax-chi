###########
# Choices #
###########


from abc import abstractmethod

import jax.lax
import jax.numpy as jnp
import jax.tree_util as jtu
import rich.tree as rich_tree
from jax.experimental import checkify
from plum import dispatch

import genjax._src.core.pretty_printing as gpp
from genjax._src.checkify import optional_check
from genjax._src.core.datatypes.selection import (
    AllSelection,
    HierarchicalSelection,
    MapSelection,
    NoneSelection,
    Selection,
    TraceSlice,
)
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    BoolArray,
    Callable,
    IntArray,
    List,
    TraceSliceComponent,
    Tuple,
    typecheck,
)


class Choice(Pytree):
    """`Choice` is the abstract base class of the type of random choices.

    The type `Choice` denotes a value which can be sampled from a generative function. There are many instances of `Choice` - distributions, for instance, utilize `ChoiceValue` - an implementor of `Choice` which wraps a single value. Other generative functions use map-like (or dictionary-like) `ChoiceMap` instances to represent their choices.
    """


    class Filtration:
        def __init__(self, choice: "Choice"):
            self.choice = choice

        def __call__(self, selection: Selection):
            return self.choice.filter_selection(selection)

        def __getitem__(
            self, t: TraceSliceComponent | Tuple[TraceSliceComponent, ...]
        ) -> "Choice":
            if not isinstance(t, tuple):
                t = (t,)
            return self.choice.filter_selection(TraceSlice(t))

    @property
    def filter(self):
        return Choice.Filtration(self)

    @abstractmethod
    def filter_selection(self, selection: Selection) -> "Choice": ...

    @abstractmethod
    def merge(self, other: "Choice") -> Tuple["Choice", "Choice"]:
        pass

    @abstractmethod
    def get_selection(self) -> Selection:
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

    def _trace_type(self):
        """This is a lazy reference to the Trace type, to avoid a circular
        import dependency"""
        from genjax._src.core.datatypes.generative import Trace
        return Trace

    def _is_trace(self, t):
        return isinstance(t, self._trace_type())

    # Remove all trace metadata, and just return choices.
    def strip(self):
        def _inner(v):
            if self._is_trace(v):
                return v.get_choices()
            else:
                return v

        return jtu.tree_map(_inner, self, is_leaf=self._is_trace)



class EmptyChoice(Choice):
    """A `Choice` implementor which denotes an empty event."""

    def filter_selection(self, selection):
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
    def filter_selection(self, selection: AllSelection):
        return self

    @dispatch
    def filter_selection(self, selection: TraceSlice):
        """TODO(colin): I'm not sure it's principled to consider a choice value
        as being not being selected by a selection."""
        return self

    @dispatch
    def filter_selection(self, selection):
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
    def filter_selection(
        self,
        selection: AllSelection,
    ) -> "ChoiceMap":
        return self

    @dispatch
    def filter_selection(
        self,
        selection: NoneSelection,
    ) -> EmptyChoice:
        return EmptyChoice()

    @dispatch
    def filter_selection(
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
            choice = tr.strip()
            selection = genjax.select("x")
            filtered = choice.filter(selection)
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

    def filter_selection(
        self,
        selection: MapSelection | TraceSlice,
    ) -> Choice:
        if isinstance(selection, NoneSelection):
            return EmptyChoice()
        if isinstance(selection, AllSelection):
            return self

        def inner():
            for k, v in self.get_submaps_shallow():
                # TODO(colin): might make sense to give AllSelection the has_addr property
                # as well as get_subselection
                if selection.has_addr(k):
                    f = v.filter_selection(selection.get_subselection(k))
                    if not isinstance(f, EmptyChoice):
                        yield k, f

        trie = Trie(dict(inner()))

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
            if not isinstance(v, ChoiceMap) and not self._is_trace(v)
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
        assert isinstance(self.value, Choice)
        return jnp.logical_and(self.flag, self.value.is_empty())

    def filter_selection(self, selection: Selection):
        return Mask(self.flag, self.value.get_choices().filter_selection(selection))

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
