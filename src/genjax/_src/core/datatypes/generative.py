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

import functools
from abc import abstractmethod

import jax

from genjax._src.core.datatypes.choice import Choice, ChoiceMap, EmptyChoice, Mask
from genjax._src.core.datatypes.selection import (
    AllSelection,
    NoneSelection,
    Selection,
    TraceSlice,
)
from genjax._src.core.datatypes.slice import SliceCompiler
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    List,
    Optional,
    PRNGKey,
    TraceSliceComponent,
    Tuple,
    Union,
    dispatch,
    typecheck,
)

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
    def get_choices(self) -> "ChoiceMap":
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

    class Projection:
        """A Projection is a helper class for the `project` method
        of the Generative Function Interface. It is capable of
        taking a `Selection` object in function call style, as
        in `tr.project(genjax.select("x"))`, or using Trace Slice
        notation using square brackets as in `tr.project["x"]`.
        """

        def __init__(self, trace):
            self.trace: Trace = trace

        def __call__(self, selection: Selection) -> FloatArray:
            return self.trace.project_selection(selection)

        def __getitem__(
            self, selection: TraceSliceComponent | Tuple[TraceSliceComponent, ...]
        ) -> FloatArray:
            if not isinstance(selection, tuple):
                selection = (selection,)
            return self.trace.project_selection(TraceSlice(selection))

    @property
    def project(self):
        """Return the total contribution to the score of the
        addresses contained within a selection. In GenJAX, a
        selection can be made in one of two ways.

        The easiest way is to use slice notation, as in
        `trace.project['x', 'y']`. Strings as indexes in trace
        slice notation represent paths in the trace tree. For
        vector layers of the tree (as produced by combinators
        such as `map_combinator` and `unfold_combinator`) can
        be indexed with numpy-style array-slice notation.

        Another way is to create a `Selection` object, with
        `genjax.select` and related Selection subclasses. Such
        selections are passed to `project` as function arguments,
        as in `trace.project(selection)`.

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
            # Selection object method
            selection = genjax.select("x")
            x_score = tr.project(selection)
            # Slice method
            assert x_score == tr.project["x"]
            x_score_t = genjax.bernoulli.logpdf(tr["x"], 0.3)
            print(console.render((x_score_t, x_score)))
            ```
        """
        return Trace.Projection(self)

    @dispatch
    def project_selection(
        self,
        selection: NoneSelection,
    ) -> FloatArray:
        return 0.0

    @dispatch
    def project_selection(
        self,
        selection: AllSelection,
    ) -> FloatArray:
        return self.get_score()

    @dispatch
    def project_selection(self, selection: "Selection") -> FloatArray:
        raise NotImplementedError

    @dispatch
    def update(
        self,
        key: PRNGKey,
        choices: "Choice",
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

    @property
    def filter(self):
        return Choice.Filtration(self.strip())

    def merge(self, other: Choice) -> Tuple[Choice, Choice]:
        return self.strip().merge(other.strip())

    def get_selection(self):
        return self.strip().get_selection()

    def strip(self):
        return self.get_choices().strip()

    def augment(self, s: Tuple[TraceSliceComponent], value: Any) -> "AugmentedTrace":
        return AugmentedTrace(self, TraceSlice(s), value)

    class Updater:
        tr: "Trace | AugmentedTrace"

        class Setter:
            u: "Trace.Updater"
            ts: TraceSlice

            def __init__(self, u, ts):
                self.u = u
                self.ts = ts

            def set(self, value: Any):
                return AugmentedTrace(self.u.tr, self.ts, value)

        def __init__(self, tr):
            self.tr = tr

        def __getitem__(self, addr):
            if not isinstance(addr, tuple):
                addr = (addr,)
            return Trace.Updater.Setter(self, TraceSlice(addr))

    @property
    def at(self):
        return Trace.Updater(self)

    def __getitem__(self, x):
        return self.get_choices()[x]


class AugmentedTrace:
    """An AugmentedTrace holds a reference to a parent trace and a single mapping
    from a TraceSlice to a value. The `update` method "compiles" the chain of
    AugmentedTraces leading back to the root trace into the equivalent Choice
    and runs `update` on the result."""

    inner: Union["AugmentedTrace", "Trace"]
    s: TraceSlice
    value: Any
    compiler = SliceCompiler()

    def __init__(self, inner, s: TraceSlice, value):
        self.inner = inner
        self.s = s
        self.value = value
        self.choice = self.compiler.slice_to_choice(s, value)

    @property
    def at(self):
        return Trace.Updater(self)

    def augment(self, s: Tuple[TraceSliceComponent], value: Any) -> "AugmentedTrace":
        return AugmentedTrace(self, TraceSlice(s), value)

    def update(self, key: PRNGKey, argdiffs: Optional[Tuple] = None):
        # Walk back through the augmentation chain, building a list of
        # mutations to make.
        mutations: List[Choice] = []
        t: Union[Trace, AugmentedTrace] = self
        while isinstance(t, AugmentedTrace):
            mutations.append(t.choice)
            t = t.inner
        assert isinstance(t, Trace)
        # TODO(colin): merge keeps track of things that are overwritten: maybe we
        # should warn when that happens?
        choice = functools.reduce(lambda a, b: a.merge(b)[0], mutations)
        # TODO(colin): maybe remove @dispatch on Trace.update and do it this way instead
        if argdiffs is not None:
            return t.update(key, choice, argdiffs)
        else:
            return t.update(key, choice)


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
