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

import jax
import jax.tree_util as jtu
import numpy as np
import rich

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.masking import Mask
from genjax._src.core.datatypes.tracetypes import Bottom
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.datatypes.tree import Leaf
from genjax._src.core.datatypes.tree import Tree
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.staging import is_concrete
from genjax._src.core.pytree import Pytree
from genjax._src.core.pytree import tree_grad_split
from genjax._src.core.pytree import tree_zipper
from genjax._src.core.typing import Any
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


#################################
# Abstract generative datatypes #
#################################

#####
# ChoiceMap
#####


@dataclasses.dataclass
class ChoiceMap(Tree):
    @abc.abstractmethod
    def is_empty(self) -> BoolArray:
        pass

    @abc.abstractmethod
    def merge(self, other: "ChoiceMap") -> Tuple["ChoiceMap", "ChoiceMap"]:
        pass

    def get_selection(self) -> "Selection":
        """Convert a `ChoiceMap` to a `Selection`."""
        raise Exception(
            f"`get_selection` is not implemented for choice map of type {type(self)}",
        )

    def safe_merge(self, other: "ChoiceMap") -> "ChoiceMap":
        new, discard = self.merge(other)
        assert discard.is_empty()
        return new

    def unsafe_merge(self, other: "ChoiceMap") -> "ChoiceMap":
        new, discard = self.merge(other)
        return new

    def get_choices(self):
        return self

    def strip(self):
        def _check(v):
            return isinstance(v, Trace)

        def _inner(v):
            if isinstance(v, Trace):
                return v.strip()
            else:
                return v

        return jtu.tree_map(_inner, self, is_leaf=_check)

    ###########
    # Dunders #
    ###########

    def __eq__(self, other):
        return self.flatten() == other.flatten()

    def __getitem__(self, addr):
        choice = self.get_subtree(addr)
        if isinstance(choice, Leaf):
            v = choice.get_leaf_value()
            return v
        else:
            return choice

    # Optional: mutable setter.
    def __setitem__(self, key, value):
        raise Exception(
            f"ChoiceMap of type {type(self)} does not implement __setitem__.",
        )

    def __add__(self, other):
        return self.safe_merge(other)

    ###################
    # Pretty printing #
    ###################

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = rich.tree.Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


#####
# Trace
#####


@dataclasses.dataclass
class Trace(ChoiceMap, Tree):
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
            v = chm.get_leaf_value()
            print(console.render((retval, v)))
            ```
        """

    @abc.abstractmethod
    def get_score(self) -> FloatArray:
        """Return the joint log score of the `Trace`.

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

    @abc.abstractmethod
    def project(self, selection: "Selection") -> FloatArray:
        """Given a `Selection`, return the total contribution to the joint log
        score of the addresses contained within the `Selection`.

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

    def get_aux(
        self,
    ) -> Tuple:
        raise NotImplementedError

    def update(self, key, choices, argdiffs):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, choices, argdiffs)

    #################################
    # Default choice map interfaces #
    #################################

    def is_empty(self):
        return self.strip().is_empty()

    def merge(self, other: ChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return self.strip().merge(other.strip())

    def has_subtree(self, addr) -> BoolArray:
        choices = self.get_choices()
        return choices.has_subtree(addr)

    def get_subtree(self, addr) -> ChoiceMap:
        choices = self.get_choices()
        return choices.get_subtree(addr)

    def get_subtrees_shallow(self):
        choices = self.get_choices()
        return choices.get_subtrees_shallow()

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

    def __getitem__(self, addr):
        choices = self.get_choices()
        choice = choices.get_subtree(addr)
        if isinstance(choice, Mask):
            if is_concrete(choice.mask):
                if choice.mask:
                    return choice.unmask()
                else:
                    return EmptyChoiceMap()
            else:
                return choice
        elif isinstance(choice, Leaf):
            return choice.get_leaf_value()
        else:
            return choice


#####
# Selection
#####


@dataclasses.dataclass
class Selection(Tree):
    @abc.abstractmethod
    def filter(self, chm: ChoiceMap) -> ChoiceMap:
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
            filtered = selection.filter(chm)
            print(console.render(filtered))
            ```
        """

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
            filtered = complement.filter(chm)
            print(console.render(filtered))
            ```
        """

    def get_selection(self):
        return self

    def __getitem__(self, addr):
        subselection = self.get_subtree(addr)
        return subselection


#####
# Generative function
#####


@dataclasses.dataclass
class GenerativeFunction(Pytree):
    """> Abstract base class for generative functions.

    !!! info "Interaction with JAX"

        Concrete implementations of `GenerativeFunction` will likely interact with the JAX tracing machinery if used with the languages exposed by `genjax`. Hence, there are specific implementation requirements which are more stringent than the requirements
        enforced in other Gen implementations (e.g. Gen in Julia).

        * For broad compatibility, the implementation of the interfaces *should* be compatible with JAX tracing.
        * If a user wishes to implement a generative function which is not compatible with JAX tracing, that generative function may invoke other JAX compat generative functions, but likely cannot be invoked inside of JAX compat generative functions.

    Aside from JAX compatibility, an implementor *should* match the interface signatures documented below. This is not statically checked - but failure to do so
    will lead to unintended behavior or errors.
    """

    def get_trace_type(self, *args, **kwargs) -> TraceType:
        shape = kwargs.get("shape", ())
        return Bottom(shape)

    def get_trace_class(self):
        raise NotImplementedError

    def unflatten_aux(self, interface_data: Tuple, aux_data: Tuple):
        trace_class = self.get_trace_class()
        return trace_class.unflatten_aux(interface_data, aux_data)

    @abc.abstractmethod
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        """> Given a `PRNGKey` and arguments, execute the generative function,
        returning a new `PRNGKey` and a trace.

        `simulate` can be informally thought of as forward sampling: given `key: PRNGKey` and arguments `args: Tuple`, the generative function should sample a choice map $c \sim p(\cdot; \\text{args})$, as well as any untraced randomness $r \sim p(\cdot; \\text{args}, c)$.

        The implementation of `simulate` should then create a trace holding the choice map, as well as the score $\log \\frac{p(c; \\text{args})}{q(r; \\text{args}, c)}$.

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

            Here's a slightly more complicated example using the `Hierarchical` generative function language. You can find more examples on the `Hierarchical` language page.

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

    def propose(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, ChoiceMap]]:
        """> Given a `PRNGKey` and arguments, execute the generative function,
        returning a new `PRNGKey` and a tuple containing the return value from
        the generative function call, the score of the choice map assignment,
        and the choice map.

        The default implementation just calls `simulate`, and then extracts the data from the `Trace` returned by `simulate`. Custom generative functions can overload the implementation for their own uses (e.g. if they don't have an associated `Trace` datatype, but can be uses as a proposal).

        Arguments:
            key: A `PRNGKey`.
            args: Arguments to the generative function.

        Returns:
            tup: A tuple `(retval, w, chm)` where `retval` is the return value from the generative function invocation, `w` is the log joint density (or an importance weight estimate, in the case where there is untraced randomness), and `chm` is the choice map assignment from the invocation.

        Examples:

            Here's an example using a `genjax` distribution (`normal`). Distributions are generative functions, so they support the interface.

            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            console = genjax.pretty()

            key = jax.random.PRNGKey(314159)
            key, (r, w, chm) = genjax.normal.propose(key, (0.0, 1.0))
            print(console.render(chm))
            ```

            Here's a slightly more complicated example using the `Hierarchical` generative function language. You can find more examples on the `Hierarchical` language page.

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
            key, (r, w, chm) = model.propose(key, ())
            print(console.render(chm))
            ```
        """
        tr = self.simulate(key, args)
        chm = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        return (retval, score, chm)

    @abc.abstractmethod
    def importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Trace]:
        """> Given a `PRNGKey`, a choice map (constraints), and arguments,
        execute the generative function, returning a new `PRNGKey`, a single-
        sample importance weight estimate of the conditional density evaluated
        at the non-constrained choices, and a trace whose choice map is
        consistent with the constraints.

        Arguments:
            key: A `PRNGKey`.
            args: Arguments to the generative function.

        Returns:
            key: A new (deterministically evolved) `PRNGKey`.
            tup: A tuple `(w, tr)` where `w` is an importance weight estimate of the conditional density, and `tr` is a trace capturing the data and inference data associated with the generative function invocation.
        """

    @abc.abstractmethod
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        new: ChoiceMap,
        diffs: Tuple,
    ) -> Tuple[Any, FloatArray, Trace, ChoiceMap]:
        pass

    @abc.abstractmethod
    def assess(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Any, FloatArray]:
        pass


@dataclasses.dataclass
class JAXGenerativeFunction(GenerativeFunction, Pytree):
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
            _, (_, score) = self.assess(key, merged, args)
            return score

        def retval(differentiable: Tuple, nondifferentiable: Tuple) -> Any:
            provided, args = tree_zipper(differentiable, nondifferentiable)
            merged = fixed.safe_merge(provided)
            _, (retval, _) = self.assess(key, merged, args)
            return retval

        return score, retval

    # A higher-level gradient API - it relies upon `unzip`,
    # but provides convenient access to first-order gradients.
    def choice_grad(self, key, trace, selection):
        fixed = selection.complement().filter(trace.strip())
        chm = selection.filter(trace.strip())
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
class EmptyChoiceMap(ChoiceMap, Leaf):
    def flatten(self):
        return (), ()

    def is_empty(self):
        return True

    def get_subtree(self, addr):
        return self

    def get_leaf_value(self):
        raise Exception("EmptyChoiceMap has no leaf value.")

    def set_leaf_value(self, v):
        raise Exception("EmptyChoiceMap has no leaf value.")

    def get_selection(self):
        return NoneSelection()

    def merge(self, other: ChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return other, self

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        tree = rich.tree.Tree("[bold](Empty)")
        return tree


@dataclasses.dataclass
class ValueChoiceMap(ChoiceMap, Leaf):
    value: Any

    def flatten(self):
        return (self.value,), ()

    @classmethod
    def new(cls, v):
        if isinstance(v, ValueChoiceMap):
            return ValueChoiceMap.new(v.get_leaf_value())
        else:
            return ValueChoiceMap(v)

    def is_empty(self):
        return False

    def get_leaf_value(self):
        return self.value

    def set_leaf_value(self, v):
        return ValueChoiceMap(v)

    def get_selection(self):
        return AllSelection()

    @dispatch
    def merge(self, other: "ValueChoiceMap") -> Tuple[ChoiceMap, ChoiceMap]:
        return other, self

    ###########
    # Dunders #
    ###########

    def __hash__(self):
        if isinstance(self.value, np.ndarray):
            return hash(self.value.tobytes())
        else:
            return hash(self.value)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        if isinstance(self.value, Pytree):
            return self.value.__rich_tree__(tree)
        else:
            sub_tree = gpp.tree_pformat(self.value)
            tree.add(sub_tree)
            return tree


@dataclasses.dataclass
class HierarchicalChoiceMap(ChoiceMap):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @typecheck
    @classmethod
    def new(cls, constraints: Dict):
        assert isinstance(constraints, Dict)
        trie = Trie.new()
        for (k, v) in constraints.items():
            v = (
                ValueChoiceMap(v)
                if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
                else v
            )
            trie.trie_insert(k, v)
        return HierarchicalChoiceMap(trie)

    def is_empty(self):
        return self.trie.is_empty()

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return EmptyChoiceMap()
        else:
            submap = value.get_choices()
            if isinstance(submap, Trie):
                return HierarchicalChoiceMap(submap)
            else:
                return submap

    def get_subtrees_shallow(self):
        return map(
            lambda v: (v[0], v[1].get_choices()),
            self.trie.get_subtrees_shallow(),
        )

    def get_selection(self):
        trie = Trie.new()
        for (k, v) in self.get_subtrees_shallow():
            trie[k] = v.get_selection()
        return HierarchicalSelection(trie)

    @dispatch
    def merge(self, other: "HierarchicalChoiceMap"):
        new_inner, discard = self.trie.merge(other.trie)
        return HierarchicalChoiceMap(new_inner), HierarchicalChoiceMap(discard)

    @dispatch
    def merge(self, other: EmptyChoiceMap):
        return self, EmptyChoiceMap()

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
            ValueChoiceMap(v)
            if not isinstance(v, ChoiceMap) and not isinstance(v, Trace)
            else v
        )
        self.trie.trie_insert(k, v)

    def __hash__(self):
        return hash(self.trie)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        for (k, v) in self.get_subtrees_shallow():
            subk = tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        return tree


#######################
# Concrete selections #
#######################


@dataclasses.dataclass
class NoneSelection(Selection, Leaf):
    def flatten(self):
        return (), ()

    def filter(self, v: Union[Trace, ChoiceMap]) -> ChoiceMap:
        return EmptyChoiceMap()

    def complement(self):
        return AllSelection()

    def get_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf choice value."
        )

    def set_leaf_value(self):
        raise Exception(
            "NoneSelection is a Selection: it does not provide a leaf choice value."
        )


@dataclasses.dataclass
class AllSelection(Selection, Leaf):
    def flatten(self):
        return (), ()

    def filter(self, v: Union[Trace, ChoiceMap]):
        return v.get_choices()

    def complement(self):
        return NoneSelection()

    def get_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf choice value."
        )

    def set_leaf_value(self):
        raise Exception(
            "AllSelection is a Selection: it does not provide a leaf choice value."
        )


@dataclasses.dataclass
class HierarchicalSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @typecheck
    @classmethod
    def new(cls, *addrs):
        trie = Trie.new()
        for addr in addrs:
            trie[addr] = AllSelection()
        return HierarchicalSelection(trie)

    @typecheck
    @classmethod
    def with_selections(cls, selections: Dict):
        assert isinstance(selections, Dict)
        trie = Trie.new()
        for (k, v) in selections.items():
            assert isinstance(v, Selection)
            trie.trie_insert(k, v)
        return HierarchicalSelection(trie)

    def filter(self, tree):
        def _inner(k, v):
            sub = self.trie[k]
            if sub is None:
                sub = NoneSelection()

            # Handles hierarchical in Trie.
            elif isinstance(sub, Trie):
                sub = HierarchicalSelection(sub)

            under = sub.filter(v)
            return k, under

        trie = Trie.new()
        iter = tree.get_subtrees_shallow()
        for (k, v) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        if isinstance(tree, TraceType):
            return type(tree)(trie, tree.get_rettype())
        else:
            return HierarchicalChoiceMap(trie)

    def complement(self):
        return HierarchicalComplementSelection(self.trie)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return NoneSelection()
        else:
            return value

    def get_subtrees_shallow(self):
        return self.trie.get_subtrees_shallow()


@dataclasses.dataclass
class HierarchicalComplementSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.selection,), ()

    def filter(self, chm):
        def _inner(k, v):
            sub = self.trie[k]
            if sub is None:
                sub = NoneSelection()

            # Handles hierarchical in Trie.
            elif isinstance(sub, Trie):
                sub = HierarchicalSelection(sub)

            under = sub.complement().filter(v)
            return k, under

        trie = Trie.new()
        iter = chm.get_subtrees_shallow()
        for (k, v) in map(lambda args: _inner(*args), iter):
            if not isinstance(v, EmptyChoiceMap):
                trie[k] = v

        if isinstance(chm, TraceType):
            return type(chm)(trie, chm.get_rettype())
        else:
            return HierarchicalChoiceMap(trie)

    def complement(self):
        return HierarchicalSelection(self.trie)

    def has_subtree(self, addr):
        return self.trie.has_subtree(addr)

    def get_subtree(self, addr):
        value = self.trie.get_subtree(addr)
        if value is None:
            return NoneSelection()
        else:
            return value

    def get_subtrees_shallow(self):
        return self.trie.get_subtrees_shallow()


##############
# Shorthands #
##############

empty_choice_map = EmptyChoiceMap.new
emp_chm = empty_choice_map
value_choice_map = ValueChoiceMap.new
val_chm = value_choice_map
all_select = AllSelection.new
all_sel = all_select
none_select = NoneSelection.new
none_sel = none_select
choice_map = HierarchicalChoiceMap.new
select = HierarchicalSelection.new
