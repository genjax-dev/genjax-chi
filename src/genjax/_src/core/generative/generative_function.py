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
from typing import TYPE_CHECKING

from genjax._src.core.generative.choice_map import (
    ChoiceMap,
    ChoiceMapConstraint,
)
from genjax._src.core.generative.core import (
    Argdiffs,
    Arguments,
    Constraint,
    EditRequest,
    Projection,
    Retdiff,
    Sample,
    Score,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import empty_trace
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    InAxes,
    Int,
    PRNGKey,
    Self,
    String,
    TypeVar,
)

# Import `genjax` so static typecheckers can see the circular reference to "genjax.ChoiceMap" below.
if TYPE_CHECKING:
    import genjax

_C = TypeVar("_C", bound=Callable[..., Any])
ArgTuple = TypeVar("ArgTuple", bound=tuple[Any, ...])

# Generative Function type variables
R = TypeVar("R")
"""
Generic denoting the return type of a generative function.
"""
S = TypeVar("S")

Carry = TypeVar("Carry")
Y = TypeVar("Y")


#########
# Trace #
#########


class Trace(Generic[R], Pytree):
    """
    `Trace` is the type of traces of generative functions.

    A trace is a data structure used to represent sampled executions of
    generative functions. Traces track metadata associated with the probabilities
    of choices, as well as other data associated with
    the invocation of a generative function, including the arguments it
    was invoked with, its return value, and the identity of the generative function itself.
    """

    @abstractmethod
    def get_args(self) -> Arguments:
        """Returns the [`Arguments`][genjax.core.Arguments] for the [`GenerativeFunction`][genjax.core.GenerativeFunction] invocation which created the [`Trace`][genjax.core.Trace]."""

    @abstractmethod
    def get_retval(self) -> R:
        """Returns the `R` from the [`GenerativeFunction`][genjax.core.GenerativeFunction] invocation which created the [`Trace`][genjax.core.Trace]."""

    @abstractmethod
    def get_score(self) -> Score:
        """Return the [`Score`][genjax.core.Score] of the `Trace`.

        The score must satisfy a particular mathematical specification: it's either an exact density evaluation of $P$ (the distribution over samples) for the sample returned by [`genjax.Trace.get_sample`][], or _a sample from an estimator_ (a density estimate) if the generative function contains _untraced randomness_.

        Let $s$ be the score, $t$ the sample, and $a$ the arguments: when the generative function contains no _untraced randomness_, the score (in logspace) is given by:

        $$
        \\log s := \\log P(t; a)
        $$

        (**With untraced randomness**) Gen allows for the possibility of sources of randomness _which are not traced_. When these sources are included in generative computations, the score is defined so that the following property holds:

        $$
        \\mathbb{E}_{r\\sim~P(r | t; a)}\\big[\\frac{1}{s}\\big] = \\frac{1}{P(t; a)}
        $$

        This property is the one you'd want to be true if you were using a generative function with untraced randomness _as a proposal_ in a routine which uses importance sampling, for instance.

        In GenJAX, one way you might encounter this is by using pseudo-random routines in your modeling code:
        ```python
        # notice how the key is explicit
        @genjax.gen
        def model_with_untraced_randomness(key: PRNGKey):
            x = genjax.normal(0.0, 1.0) "x"
            v = some_random_process(key, x)
            y = genjax.normal(v, 1.0) @ "y"
        ```

        In this case, the score (in logspace) is given by:

        $$
        \\log s := \\log P(r, t; a) - \\log Q(r; a)
        $$

        which satisfies the requirement by virtue of the fact:

        $$
        \\begin{aligned}
        \\mathbb{E}_{r\\sim~P(r | t; a)}\\big[\\frac{1}{s}\\big] &= \\mathbb{E}_{r\\sim P(r | t; a)}\\big[\\frac{Q(r; a)}{P(r, t; a)} \\big] \\\\ &= \\frac{1}{P(t; a)} \\mathbb{E}_{r\\sim P(r | t; a)}\\big[\\frac{Q(r; a)}{P(r | t; a)}\\big] \\\\
        &= \\frac{1}{P(t; a)}
        \\end{aligned}
        $$

        """

    @abstractmethod
    def get_sample(self) -> Sample:
        """Return the [`Sample`][genjax.core.Sample] sampled from the distribution over samples by the generative function during the invocation which created the [`Trace`][genjax.core.Trace]."""

    @abstractmethod
    def get_choices(self) -> "genjax.ChoiceMap":
        """Version of [`genjax.Trace.get_sample`][] for traces where the sample is an instance of [`genjax.ChoiceMap`][]."""
        pass

    @abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction[R]":
        """Returns the [`GenerativeFunction`][genjax.core.GenerativeFunction] whose invocation created the [`Trace`][genjax.core.Trace]."""
        pass

    def edit(
        self,
        key: PRNGKey,
        request: EditRequest,
        argdiffs: tuple[Any, ...] | None = None,
    ) -> tuple[Self, Weight, Retdiff[R], EditRequest]:
        """
        This method calls out to the underlying [`GenerativeFunction.edit`][genjax.core.GenerativeFunction.edit] method - see [`EditRequest`][genjax.core.EditRequest] and [`edit`][genjax.core.GenerativeFunction.edit] for more information.
        """
        return request.edit(
            key,
            self,
            Diff.tree_diff_no_change(self.get_args()) if argdiffs is None else argdiffs,
        )  # pyright: ignore[reportReturnType]

    def update(
        self,
        key: PRNGKey,
        constraint: ChoiceMap,
        argdiffs: tuple[Any, ...] | None = None,
    ) -> tuple[Self, Weight, Retdiff[R], Constraint]:
        """
        This method calls out to the underlying [`GenerativeFunction.edit`][genjax.core.GenerativeFunction.edit] method - see [`EditRequest`][genjax.core.EditRequest] and [`edit`][genjax.core.GenerativeFunction.edit] for more information.
        """
        return self.get_gen_fn().update(
            key,
            self,
            constraint,
            Diff.tree_diff_no_change(self.get_args()) if argdiffs is None else argdiffs,
        )  # pyright: ignore[reportReturnType]

    def project(
        self,
        key: PRNGKey,
        projection: Projection[Any],
    ) -> Weight:
        gen_fn = self.get_gen_fn()
        return gen_fn.project(
            key,
            self,
            projection,
        )

    ###################
    # Batch semantics #
    ###################

    @property
    def batch_shape(self):
        return len(self.get_score())


#######################
# Generative function #
#######################


class GenerativeFunction(Generic[R], Pytree):
    """
    `GenerativeFunction` is the type of _generative functions_, the main computational object in Gen.

    Generative functions are a type of probabilistic program. In terms of their mathematical specification, they come equipped with a few ingredients:

    * (**Distribution over samples**) $P(\\cdot_t, \\cdot_r; a)$ - a probability distribution over samples $t$ and untraced randomness $r$, indexed by arguments $a$. This ingredient is involved in all the interfaces and specifies the distribution over samples which the generative function represents.
    * (**Family of K/L proposals**) $(K(\\cdot_t, \\cdot_{K_r}; u, t), L(\\cdot_t, \\cdot_{L_r}; u, t)) = \\mathcal{F}(u, t)$ - a family of pairs of probabilistic programs (referred to as K and L), indexed by [`EditRequest`][genjax.core.EditRequest] $u$ and an existing sample $t$. This ingredient supports the [`edit`][genjax.core.GenerativeFunction.edit] and [`importance`][genjax.core.GenerativeFunction.importance] interface, and is used to specify an SMCP3 move which the generative function must provide in response to an edit request. K and L must satisfy additional properties, described further in [`edit`][genjax.core.GenerativeFunction.edit].
    * (**Return value function**) $f(t, r, a)$ - a deterministic return value function, which maps samples and untraced randomness to return values.

    Generative functions also support a family of [`Target`][genjax.inference.Target] distributions - a [`Target`][genjax.inference.Target] distribution is a (possibly unnormalized) distribution, typically induced by inference problems.

    * $\\delta_\\emptyset$ - the empty target, whose only possible value is the empty sample, with density 1.
    * (**Family of targets induced by $P$**) $T_P(a, c)$ - a family of targets indexed by arguments $a$ and [`Constraint`][genjax.core.Constraint] $c$, created by pairing the distribution over samples $P$ with arguments and constraint.

    Generative functions expose computations using these ingredients through the _generative function interface_ (the methods which are documented below).

    Examples:
        The interface methods can be used to implement inference algorithms directly - here's a simple example using bootstrap importance sampling directly:
        ```python exec="yes" html="true" source="material-block" session="core"
        import jax
        from jax.scipy.special import logsumexp
        from jax.random import PRNGKey
        import jax.tree_util as jtu
        from genjax import ChoiceMapBuilder as C
        from genjax import gen, uniform, flip, categorical


        @gen
        def model():
            p = uniform(0.0, 1.0) @ "p"
            f1 = flip(p) @ "f1"
            f2 = flip(p) @ "f2"


        # Bootstrap importance sampling.
        def importance_sampling(key, constraint):
            key, sub_key = jax.random.split(key)
            sub_keys = jax.random.split(sub_key, 5)
            tr, log_weights = jax.vmap(model.importance, in_axes=(0, None, None))(
                sub_keys, constraint, ()
            )
            logits = log_weights - logsumexp(log_weights)
            idx = categorical(logits)(key)
            return jtu.tree_map(lambda v: v[idx], tr.get_sample())


        sub_keys = jax.random.split(PRNGKey(0), 50)
        samples = jax.jit(jax.vmap(importance_sampling, in_axes=(0, None)))(
            sub_keys, C.kw(f1=True, f2=True)
        )
        print(samples.render_html())
        ```
    """

    def __call__(self, *args, **kwargs) -> "GenerativeFunctionClosure[R]":
        return GenerativeFunctionClosure(self, args, kwargs)

    def __abstract_call__(self, *args) -> R:
        """Used to support JAX tracing, although this default implementation involves no
        JAX operations (it takes a fixed-key sample from the return value).

        Generative functions may customize this to improve compilation time.
        """
        return self.get_zero_trace(*args).get_retval()

    def handle_kwargs(self) -> "GenerativeFunction[R]":
        return IgnoreKwargs(self)

    def get_zero_trace(self, *args, **_kwargs) -> Trace[R]:
        """
        Returns a trace with zero values for all leaves, generated without executing the generative function.

        This method is useful for static analysis and shape inference without executing the generative function. It returns a trace with the same structure as a real trace, but filled with zero or default values.

        Args:
            *args: The arguments to the generative function.
            **_kwargs: Ignored keyword arguments.

        Returns:
            A trace with zero values, matching the structure of a real trace.

        Note:
            This method uses the `empty_trace` utility function, which creates a trace without spending any FLOPs. The resulting trace has the correct structure but contains placeholder zero values.

        Example:
            ```python exec="yes" html="true" source="material-block" session="core"
            @genjax.gen
            def weather_model():
                temperature = genjax.normal(20.0, 5.0) @ "temperature"
                is_sunny = genjax.bernoulli(0.7) @ "is_sunny"
                return {"temperature": temperature, "is_sunny": is_sunny}


            zero_trace = weather_model.get_zero_trace()
            print("Zero trace structure:")
            print(zero_trace.render_html())

            print("\nActual simulation:")
            key = jax.random.PRNGKey(0)
            actual_trace = weather_model.simulate(key, ())
            print(actual_trace.render_html())
            ```
        """
        return empty_trace(self, args)

    @abstractmethod
    def simulate(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Trace[R]:
        """
        Execute the generative function, sampling from its distribution over samples, and return a [`Trace`][genjax.core.Trace].

        ## More on traces

        The [`Trace`][genjax.core.Trace] returned by `simulate` implements its own interface.

        It is responsible for storing the arguments of the invocation ([`genjax.Trace.get_args`][]), the return value of the generative function ([`genjax.Trace.get_retval`][]), the identity of the generative function which produced the trace ([`genjax.Trace.get_gen_fn`][]), the sample of traced random choices produced during the invocation ([`genjax.Trace.get_sample`][]) and _the score_ of the sample ([`genjax.Trace.get_score`][]).

        Examples:
            ```python exec="yes" html="true" source="material-block" session="core"
            import genjax
            from jax import vmap, jit
            from jax.random import PRNGKey
            from jax.random import split


            @genjax.gen
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                return x


            key = PRNGKey(0)
            tr = model.simulate(key, ())
            print(tr.render_html())
            ```

            Another example, using the same model, composed into [`genjax.repeat`](combinators.md#genjax.repeat) - which creates a new generative function, which has the same interface:
            ```python exec="yes" html="true" source="material-block" session="core"
            @genjax.gen
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                return x


            key = PRNGKey(0)
            tr = model.repeat(n=10).simulate(key, ())
            print(tr.render_html())
            ```

            (**Fun, flirty, fast ... parallel?**) Feel free to use `jax.jit` and `jax.vmap`!
            ```python exec="yes" html="true" source="material-block" session="core"
            key = PRNGKey(0)
            sub_keys = split(key, 10)
            sim = model.repeat(n=10).simulate
            tr = jit(vmap(sim, in_axes=(0, None)))(sub_keys, ())
            print(tr.render_html())
            ```
        """

    @abstractmethod
    def assess(
        self,
        sample: ChoiceMap,
        args: Arguments,
    ) -> tuple[Score, R]:
        """
        Return [the score][genjax.core.Trace.get_score] and [the return value][genjax.core.Trace.get_retval] when the generative function is invoked with the provided arguments, and constrained to take the provided sample as the sampled value.

        It is an error if the provided sample value is off the support of the distribution over the `Sample` type, or otherwise induces a partial constraint on the execution of the generative function (which would require the generative function to provide an `edit` implementation which responds to the `EditRequest` induced by the [`importance`][genjax.core.GenerativeFunction.importance] interface).

        Examples:
            This method is similar to density evaluation interfaces for distributions.
            ```python exec="yes" html="true" source="material-block" session="core"
            from genjax import normal
            from genjax import ChoiceMapBuilder as C

            sample = C.v(1.0)
            score, retval = normal.assess(sample, (1.0, 1.0))
            print((score, retval))
            ```

            But it also works with generative functions that sample from spaces with more structure:

            ```python exec="yes" html="true" source="material-block" session="core"
            from genjax import gen
            from genjax import normal
            from genjax import ChoiceMapBuilder as C


            @gen
            def model():
                v1 = normal(0.0, 1.0) @ "v1"
                v2 = normal(v1, 1.0) @ "v2"


            sample = C.kw(v1=1.0, v2=0.0)
            score, retval = model.assess(sample, ())
            print((score, retval))
            ```
        """

    @abstractmethod
    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Arguments,
    ) -> tuple[Trace[R], Weight]:
        pass

    @abstractmethod
    def project(
        self,
        key: PRNGKey,
        trace: Trace[R],
        projection: Projection[Any],
    ) -> Weight:
        pass

    @abstractmethod
    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        """
        Update a trace in response to an [`EditRequest`][genjax.core.EditRequest], returning a new [`Trace`][genjax.core.Trace], an incremental [`Weight`][genjax.core.Weight] for the new target, a [`Retdiff`][genjax.core.Retdiff] return value tagged with change information, and a backward [`EditRequest`][genjax.core.EditRequest] which requests the reverse move (to go back to the original trace).

        The specification of this interface is parametric over the kind of `EditRequest` -- responding to an `EditRequest` instance requires that the generative function provides an implementation of a sequential Monte Carlo move in the [SMCP3](https://proceedings.mlr.press/v206/lew23a.html) framework. Users of inference algorithms are not expected to understand the ingredients, but inference algorithm developers are.

        Examples:
            Updating a trace in response to a request for a [`Target`][genjax.inference.Target] change induced by a change to the arguments:
            ```python exec="yes" source="material-block" session="core"
            from genjax import gen
            from genjax import normal
            from genjax import Diff
            from genjax import Update
            from genjax import ChoiceMapConstraint
            from genjax import ChoiceMap as C

            key = PRNGKey(0)


            @gen
            def model(var):
                v1 = normal(0.0, 1.0) @ "v1"
                v2 = normal(v1, var) @ "v2"
                return v2


            # Generating an initial trace properly weighted according
            # to the target induced by the constraint.
            constraint = C.kw(v2=1.0)
            initial_tr, w = model.importance(key, constraint, (1.0,))

            # Updating the trace to a new target.
            new_tr, inc_w, retdiff, bwd_prob = model.edit(
                key,
                initial_tr,
                Update(
                    ChoiceMapConstraint(C.empty()),
                ),
                Diff.unknown_change((3.0,)),
            )
            ```

            Now, let's inspect the trace:
            ```python exec="yes" html="true" source="material-block" session="core"
            # Inspect the trace, the sampled values should not have changed!
            sample = new_tr.get_sample()
            print(sample["v1"], sample["v2"])
            ```

            And the return value diff:
            ```python exec="yes" html="true" source="material-block" session="core"
            # The return value also should not have changed!
            print(retdiff.render_html())
            ```

            As expected, neither have changed -- but the weight is non-zero:
            ```python exec="yes" html="true" source="material-block" session="core"
            print(w)
            ```

        ## Mathematical ingredients behind edit

        The `edit` interface exposes [SMCP3 moves](https://proceedings.mlr.press/v206/lew23a.html). Here, we omit the measure theoretic description, and refer interested readers to [the paper](https://proceedings.mlr.press/v206/lew23a.html). Informally, the ingredients of such a move are:

        * The previous target $T$.
        * The new target $T'$.
        * A pair of kernel probabilistic programs, called $K$ and $L$:
            * The K kernel is a kernel probabilistic program which accepts a previous sample $x_{t-1}$ from $T$ as an argument, may sample auxiliary randomness $u_K$, and returns a new sample $x_t$ approximately distributed according to $T'$, along with transformed randomness $u_L$.
            * The L kernel is a kernel probabilistic program which accepts the new sample $x_t$, and provides a density evaluator for the auxiliary randomness $u_L$ which K returns, and an inverter $x_t \\mapsto x_{t-1}$ which is _almost everywhere_ the identity function.

        The specification of these ingredients are encapsulated in the type signature of the `edit` interface.

        ## Understanding the `edit` interface

        The `edit` interface uses the mathematical ingredients described above to perform probability-aware mutations and incremental [`Weight`][genjax.core.Weight] computations on [`Trace`][genjax.core.Trace] instances, which allows Gen to provide automation to support inference agorithms like importance sampling, SMC, MCMC and many more.

        An `EditRequest` denotes a function $tr \\mapsto (T, T')$ from traces to a pair of targets (the previous [`Target`][genjax.inference.Target] $T$, and the final [`Target`][genjax.inference.Target] $T'$).

        Several common types of moves can be requested via the `Update` type:

        ```python exec="yes" source="material-block" session="core"
        from genjax import Update
        from genjax import ChoiceMap, ChoiceMapConstraint

        g = Update(
            ChoiceMapConstraint(ChoiceMap.empty()),  # Constraint
        )
        ```

        `Update` contains information about changes to the arguments of the generative function ([`Argdiffs`][genjax.core.Argdiffs]) and a constraint which specifies an additional move to be performed.

        ```python exec="yes" html="true" source="material-block" session="core"
        new_tr, inc_w, retdiff, bwd_prob = model.edit(
            key,
            initial_tr,
            Update(
                ChoiceMapConstraint(C.kw(v1=3.0)),
            ),
            Diff.unknown_change((3.0,)),
        )
        print((new_tr.get_sample()["v1"], w))
        ```

        **Additional notes on [`Argdiffs`][genjax.core.Argdiffs]**

        Argument changes induce changes to the distribution over samples, internal K and L proposals, and (by virtue of changes to $P$) target distributions. The [`Argdiffs`][genjax.core.Argdiffs] type denotes the type of values attached with a _change type_, a piece of data which indicates how the value has changed from the arguments which created the trace. Generative functions can utilize change type information to inform efficient [`edit`][genjax.core.GenerativeFunction.edit] implementations.
        """
        pass

    ######################
    # Derived interfaces #
    ######################

    def update(
        self,
        key: PRNGKey,
        trace: Trace[R],
        constraint: ChoiceMap,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], Constraint]:
        request = Update(
            ChoiceMapConstraint(constraint),
        )
        tr, w, rd, bwd = request.edit(
            key,
            trace,
            argdiffs,
        )
        assert isinstance(bwd, Update), type(bwd)
        return tr, w, rd, bwd.constraint

    def importance(
        self,
        key: PRNGKey,
        constraint: ChoiceMap | Constraint,
        args: Arguments,
    ) -> tuple[Trace[R], Weight]:
        """
        Returns a properly weighted pair, a [`Trace`][genjax.core.Trace] and a [`Weight`][genjax.core.Weight], properly weighted for the target induced by the generative function for the provided constraint and arguments.

        Examples:
            (**Full constraints**) A simple example using the `importance` interface on distributions:
            ```python exec="yes" html="true" source="material-block" session="core"
            from genjax import normal
            from genjax import ChoiceMapBuilder as C
            from jax.random import PRNGKey

            key = PRNGKey(0)

            tr, w = normal.importance(key, C.v(1.0), (0.0, 1.0))
            print(tr.get_sample().render_html())
            ```

            (**Internal proposal for partial constraints**) Specifying a _partial_ constraint on a [`StaticGenerativeFunction`][genjax.StaticGenerativeFunction]:
            ```python exec="yes" html="true" source="material-block" session="core"
            from genjax import flip, uniform, gen
            from genjax import ChoiceMapBuilder as C


            @gen
            def model():
                p = uniform(0.0, 1.0) @ "p"
                f1 = flip(p) @ "f1"
                f2 = flip(p) @ "f2"


            tr, w = model.importance(key, C.kw(f1=True, f2=True), ())
            print(tr.get_sample().render_html())
            ```

        Under the hood, creates an [`EditRequest`][genjax.core.EditRequest] which requests that the generative function respond with a move from the _empty_ trace (the only possible value for _empty_ target $\\delta_\\emptyset$) to the target induced by the generative function for constraint $C$ with arguments $a$.
        """
        return self.generate(
            key,
            constraint
            if isinstance(constraint, Constraint)
            else ChoiceMapConstraint(constraint),
            args,
        )

    def propose(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> tuple[Sample, Score, R]:
        """
        Samples a [`Sample`][genjax.core.Sample] and any untraced randomness $r$ from the generative function's distribution over samples ($P$), and returns the [`Score`][genjax.core.Score] of that sample under the distribution, and the `R` of the generative function's return value function $f(r, t, a)$ for the sample and untraced randomness.
        """
        tr = self.simulate(key, args)
        sample = tr.get_sample()
        score = tr.get_score()
        retval = tr.get_retval()
        return sample, score, retval

    ######################################################
    # Convenience: postfix syntax for combinators / DSLs #
    ######################################################

    ###############
    # Combinators #
    ###############

    # TODO think through, or note, that the R that comes out will have to be bounded by pytree.
    def vmap(self, /, *, in_axes: InAxes = 0) -> "GenerativeFunction[R]":
        """
        Returns a [`GenerativeFunction`][genjax.GenerativeFunction] that performs a vectorized map over the argument specified by `in_axes`. Traced values are nested under an index, and the retval is vectorized.

        Args:
            in_axes: Selector specifying which input arguments (or index into them) should be vectorized. Defaults to 0, i.e., the first argument. See [this link](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees) for more detail.

        Returns:
            A new [`GenerativeFunction`][genjax.GenerativeFunction] that accepts an argument of one-higher dimension at the position specified by `in_axes`.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="gen-fn"
            import jax
            import jax.numpy as jnp
            import genjax


            @genjax.gen
            def model(x):
                v = genjax.normal(x, 1.0) @ "v"
                return genjax.normal(v, 0.01) @ "q"


            vmapped = model.vmap(in_axes=0)

            key = jax.random.PRNGKey(314159)
            arr = jnp.ones(100)

            # `vmapped` accepts an array if numbers instead of the original
            # single number that `model` accepted.
            tr = jax.jit(vmapped.simulate)(key, (arr,))

            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.vmap(in_axes=in_axes)(self)

    def repeat(self, /, *, n: Int) -> "GenerativeFunction[R]":
        """
        Returns a [`genjax.GenerativeFunction`][] that samples from `self` `n` times, returning a vector of `n` results.

        The values traced by each call `gen_fn` will be nested under an integer index that matches the loop iteration index that generated it.

        This combinator is useful for creating multiple samples from `self` in a batched manner.

        Args:
            n: The number of times to sample from the generative function.

        Returns:
            A new [`genjax.GenerativeFunction`][] that samples from the original function `n` times.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="repeat"
            import genjax, jax


            @genjax.gen
            def normal_draw(mean):
                return genjax.normal(mean, 1.0) @ "x"


            normal_draws = normal_draw.repeat(n=10)

            key = jax.random.PRNGKey(314159)

            # Generate 10 draws from a normal distribution with mean 2.0
            tr = jax.jit(normal_draws.simulate)(key, (2.0,))
            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.repeat(n=n)(self)

    def scan(
        self: "GenerativeFunction[tuple[Carry, Y]]",
        /,
        *,
        n: Int | None = None,
    ) -> "GenerativeFunction[tuple[Carry, Y]]":
        """
        When called on a [`genjax.GenerativeFunction`][] of type `(c, a) -> (c, b)`, returns a new [`genjax.GenerativeFunction`][] of type `(c, [a]) -> (c, [b])` where

        - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
        - `a` may be a primitive, an array type or a pytree (container) type with array leaves
        - `b` may be a primitive, an array type or a pytree (container) type with array leaves.

        The values traced by each call to the original generative function will be nested under an integer index that matches the loop iteration index that generated it.

        For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

        When the type of `xs` in the snippet below (denoted `[a]` above) is an array type or None, and the type of `ys` in the snippet below (denoted `[b]` above) is an array type, the semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

        ```python
        def scan(f, init, xs, length=None):
            if xs is None:
                xs = [None] * length
            carry = init
            ys = []
            for x in xs:
                carry, y = f(carry, x)
                ys.append(y)
            return carry, np.stack(ys)
        ```

        Unlike that Python version, both `xs` and `ys` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays. `None` is actually a special case of this, as it represents an empty pytree.

        The loop-carried value `c` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

        Args:
            n: optional integer specifying the number of loop iterations, which (if supplied) must agree with the sizes of leading axes of the arrays in the returned function's second argument. If supplied then the returned generative function can take `None` as its second argument.

        Returns:
            A new [`genjax.GenerativeFunction`][] that takes a loop-carried value and a new input, and returns a new loop-carried value along with either `None` or an output to be collected into the second return value.

        Examples:
            Scan for 1000 iterations with no array input:
            ```python exec="yes" html="true" source="material-block" session="scan"
            import jax
            import genjax


            @genjax.gen
            def random_walk_step(prev, _):
                x = genjax.normal(prev, 1.0) @ "x"
                return x, None


            random_walk = random_walk_step.scan(n=1000)

            init = 0.5
            key = jax.random.PRNGKey(314159)

            tr = jax.jit(random_walk.simulate)(key, (init, None))
            print(tr.render_html())
            ```

            Scan across an input array:
            ```python exec="yes" html="true" source="material-block" session="scan"
            import jax.numpy as jnp


            @genjax.gen
            def add_and_square_step(sum, x):
                new_sum = sum + x
                return new_sum, sum * sum


            # notice no `n` parameter supplied:
            add_and_square_all = add_and_square_step.scan()
            init = 0.0
            xs = jnp.ones(10)

            tr = jax.jit(add_and_square_all.simulate)(key, (init, xs))

            # The retval has the final carry and an array of all `sum*sum` returned.
            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.scan(n=n)(self)

    def accumulate(self) -> "GenerativeFunction[R]":
        """
        When called on a [`genjax.GenerativeFunction`][] of type `(c, a) -> c`, returns a new [`genjax.GenerativeFunction`][] of type `(c, [a]) -> [c]` where

        - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
        - `[c]` is an array of all loop-carried values seen during iteration (including the first)
        - `a` may be a primitive, an array type or a pytree (container) type with array leaves

        All traced values are nested under an index.

        For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

        The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation (note the similarity to [`itertools.accumulate`](https://docs.python.org/3/library/itertools.html#itertools.accumulate)):

        ```python
        def accumulate(f, init, xs):
            carry = init
            carries = [init]
            for x in xs:
                carry = f(carry, x)
                carries.append(carry)
            return carries
        ```

        Unlike that Python version, both `xs` and `carries` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays.

        The loop-carried value `c` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

        Examples:
            ```python exec="yes" html="true" source="material-block" session="scan"
            import jax
            import genjax
            import jax.numpy as jnp


            @genjax.accumulate()
            @genjax.gen
            def add(sum, x):
                new_sum = sum + x
                return new_sum


            init = 0.0
            key = jax.random.PRNGKey(314159)
            xs = jnp.ones(10)

            tr = jax.jit(add.simulate)(key, (init, xs))
            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.accumulate()(self)

    def reduce(self) -> "GenerativeFunction[R]":
        """
        When called on a [`genjax.GenerativeFunction`][] of type `(c, a) -> c`, returns a new [`genjax.GenerativeFunction`][] of type `(c, [a]) -> c` where

        - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
        - `a` may be a primitive, an array type or a pytree (container) type with array leaves

        All traced values are nested under an index.

        For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

        The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation (note the similarity to [`functools.reduce`](https://docs.python.org/3/library/itertools.html#functools.reduce)):

        ```python
        def reduce(f, init, xs):
            carry = init
            for x in xs:
                carry = f(carry, x)
            return carry
        ```

        Unlike that Python version, both `xs` and `carry` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays.

        The loop-carried value `c` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

        Examples:
            sum an array of numbers:
            ```python exec="yes" html="true" source="material-block" session="scan"
            import jax
            import genjax
            import jax.numpy as jnp


            @genjax.reduce()
            @genjax.gen
            def add(sum, x):
                new_sum = sum + x
                return new_sum


            init = 0.0
            key = jax.random.PRNGKey(314159)
            xs = jnp.ones(10)

            tr = jax.jit(add.simulate)(key, (init, xs))
            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.reduce()(self)

    def iterate(
        self,
        /,
        *,
        n: Int,
    ) -> "GenerativeFunction[R]":
        """
        When called on a [`genjax.GenerativeFunction`][] of type `a -> a`, returns a new [`genjax.GenerativeFunction`][] of type `a -> [a]` where

        - `a` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
        - `[a]` is an array of all `a`, `f(a)`, `f(f(a))` etc. values seen during iteration.

        All traced values are nested under an index.

        The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

        ```python
        def iterate(f, n, init):
            input = init
            seen = [init]
            for _ in range(n):
                input = f(input)
                seen.append(input)
            return seen
        ```

        `init` may be an arbitrary pytree value, and so multiple arrays can be iterated over at once and produce multiple output arrays.

        The iterated value `a` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `a` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

        Args:
            n: the number of iterations to run.

        Examples:
            iterative addition, returning all intermediate sums:
            ```python exec="yes" html="true" source="material-block" session="scan"
            import jax
            import genjax


            @genjax.iterate(n=100)
            @genjax.gen
            def inc(x):
                return x + 1


            init = 0.0
            key = jax.random.PRNGKey(314159)

            tr = jax.jit(inc.simulate)(key, (init,))
            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.iterate(n=n)(self)

    def iterate_final(
        self,
        /,
        *,
        n: Int,
    ) -> "GenerativeFunction[R]":
        """
        Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type `a -> a` and returns a new [`genjax.GenerativeFunction`][] of type `a -> a` where

        - `a` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
        - the original function is invoked `n` times with each input coming from the previous invocation's output, so that the new function returns $f^n(a)$

        All traced values are nested under an index.

        The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

        ```python
        def iterate_final(f, n, init):
            ret = init
            for _ in range(n):
                ret = f(ret)
            return ret
        ```

        `init` may be an arbitrary pytree value, and so multiple arrays can be iterated over at once and produce multiple output arrays.

        The iterated value `a` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `a` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

        Args:
            n: the number of iterations to run.

        Examples:
            iterative addition:
            ```python exec="yes" html="true" source="material-block" session="scan"
            import jax
            import genjax


            @genjax.iterate_final(n=100)
            @genjax.gen
            def inc(x):
                return x + 1


            init = 0.0
            key = jax.random.PRNGKey(314159)

            tr = jax.jit(inc.simulate)(key, (init,))
            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.iterate_final(n=n)(self)

    def mask(self, /) -> "GenerativeFunction[genjax.Mask[R]]":
        """
        Enables dynamic masking of generative functions. Returns a new [`genjax.GenerativeFunction`][] like `self`, but which accepts an additional boolean first argument.

        If `True`, the invocation of `self` is masked, and its contribution to the score is ignored. If `False`, it has the same semantics as if one was invoking `self` without masking.

        The return value type is a `Mask`, with a flag value equal to the supplied boolean.

        Returns:
            The masked version of the original [`genjax.GenerativeFunction`][].

        Examples:
            Masking a normal draw:
            ```python exec="yes" html="true" source="material-block" session="mask"
            import genjax, jax


            @genjax.gen
            def normal_draw(mean):
                return genjax.normal(mean, 1.0) @ "x"


            masked_normal_draw = normal_draw.mask()

            key = jax.random.PRNGKey(314159)
            tr = jax.jit(masked_normal_draw.simulate)(
                key,
                (
                    False,
                    2.0,
                ),
            )
            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.mask(self)

    def or_else(self, gen_fn: "GenerativeFunction[R]", /) -> "GenerativeFunction[R]":
        """
        Returns a [`GenerativeFunction`][genjax.GenerativeFunction] that accepts

        - a boolean argument
        - an argument tuple for `self`
        - an argument tuple for the supplied `gen_fn`

        and acts like `self` when the boolean is `True` or like `gen_fn` otherwise.

        Args:
            gen_fn: called when the boolean argument is `False`.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="gen-fn"
            import jax
            import jax.numpy as jnp
            import genjax


            @genjax.gen
            def if_model(x):
                return genjax.normal(x, 1.0) @ "if_value"


            @genjax.gen
            def else_model(x):
                return genjax.normal(x, 5.0) @ "else_value"


            @genjax.gen
            def model(toss: bool):
                # Note that the returned model takes a new boolean predicate in
                # addition to argument tuples for each branch.
                return if_model.or_else(else_model)(toss, (1.0,), (10.0,)) @ "tossed"


            key = jax.random.PRNGKey(314159)

            tr = jax.jit(model.simulate)(key, (True,))

            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.or_else(self, gen_fn)

    def switch(
        self, *branches: "GenerativeFunction[R]"
    ) -> "genjax.SwitchCombinator[R]":
        """
        Given `n` [`genjax.GenerativeFunction`][] inputs, returns a new [`genjax.GenerativeFunction`][] that accepts `n+2` arguments:

        - an index in the range $[0, n+1)$
        - a tuple of arguments for `self` and each of the input generative functions (`n+1` total tuples)

        and executes the generative function at the supplied index with its provided arguments.

        If `index` is out of bounds, `index` is clamped to within bounds.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="switch"
            import jax, genjax


            @genjax.gen
            def branch_1():
                x = genjax.normal(0.0, 1.0) @ "x1"


            @genjax.gen
            def branch_2():
                x = genjax.bernoulli(0.3) @ "x2"


            switch = branch_1.switch(branch_2)

            key = jax.random.PRNGKey(314159)
            jitted = jax.jit(switch.simulate)

            # Select `branch_2` by providing 1:
            tr = jitted(key, (1, (), ()))

            print(tr.render_html())
            ```
        """
        import genjax

        return genjax.switch(self, *branches)

    def mix(self, *fns: "GenerativeFunction[R]") -> "GenerativeFunction[R]":
        """
        Takes any number of [`genjax.GenerativeFunction`][]s and returns a new [`genjax.GenerativeFunction`][] that represents a mixture model.

        The returned generative function takes the following arguments:

        - `mixture_logits`: Logits for the categorical distribution used to select a component.
        - `*args`: Argument tuples for `self` and each of the input generative functions

        and samples from `self` or one of the input generative functions based on a draw from a categorical distribution defined by the provided mixture logits.

        Args:
            *fns: Variable number of [`genjax.GenerativeFunction`][]s to be mixed with `self`.

        Returns:
            A new [`genjax.GenerativeFunction`][] representing the mixture model.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="mix"
            import jax
            import genjax


            # Define component generative functions
            @genjax.gen
            def component1(x):
                return genjax.normal(x, 1.0) @ "y"


            @genjax.gen
            def component2(x):
                return genjax.normal(x, 2.0) @ "y"


            # Create mixture model
            mixture = component1.mix(component2)

            # Use the mixture model
            key = jax.random.PRNGKey(0)
            logits = jax.numpy.array([0.3, 0.7])  # Favors component2
            trace = mixture.simulate(key, (logits, (0.0,), (7.0,)))
            print(trace.render_html())
                ```
        """
        import genjax

        return genjax.mix(self, *fns)

    def dimap(
        self,
        /,
        *,
        pre: Callable[..., ArgTuple],
        post: Callable[[ArgTuple, R], S],
        info: String | None = None,
    ) -> "GenerativeFunction[S]":
        """
        Returns a new [`genjax.GenerativeFunction`][] and applies pre- and post-processing functions to its arguments and return value.

        !!! info
            Prefer [`genjax.GenerativeFunction.map`][] if you only need to transform the return value, or [`genjax.GenerativeFunction.contramap`][] if you only need to transform the arguments.

        Args:
            pre: A callable that preprocesses the arguments before passing them to the wrapped function. Note that `pre` must return a _tuple_ of arguments, not a bare argument. Default is the identity function.
            post: A callable that postprocesses the return value of the wrapped function. Default is the identity function.
            info: An optional string providing additional information about the `dimap` operation.

        Returns:
            A new [`genjax.GenerativeFunction`][] with `pre` and `post` applied.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="dimap"
            import jax, genjax


            # Define pre- and post-processing functions
            def pre_process(x, y):
                return (x + 1, y * 2)


            def post_process(args, retval):
                return retval**2


            @genjax.gen
            def model(x, y):
                return genjax.normal(x, y) @ "z"


            dimap_model = model.dimap(
                pre=pre_process, post=post_process, info="Square of normal"
            )

            # Use the dimap model
            key = jax.random.PRNGKey(0)
            trace = dimap_model.simulate(key, (2.0, 3.0))

            print(trace.render_html())
            ```
        """
        import genjax

        return genjax.dimap(pre=pre, post=post, info=info)(self)

    def map(
        self, f: Callable[[R], S], *, info: String | None = None
    ) -> "GenerativeFunction[S]":
        """
        Specialized version of [`genjax.dimap`][] where only the post-processing function is applied.

        Args:
            f: A callable that postprocesses the return value of the wrapped function.
            info: An optional string providing additional information about the `map` operation.

        Returns:
            A [`genjax.GenerativeFunction`][] that acts like `self` with a post-processing function to its return value.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="map"
            import jax, genjax


            # Define a post-processing function
            def square(x):
                return x**2


            @genjax.gen
            def model(x):
                return genjax.normal(x, 1.0) @ "z"


            map_model = model.map(square, info="Square of normal")

            # Use the map model
            key = jax.random.PRNGKey(0)
            trace = map_model.simulate(key, (2.0,))

            print(trace.render_html())
            ```
        """
        import genjax

        return genjax.map(f=f, info=info)(self)

    def contramap(
        self, f: Callable[..., ArgTuple], *, info: String | None = None
    ) -> "GenerativeFunction[R]":
        """
        Specialized version of [`genjax.GenerativeFunction.dimap`][] where only the pre-processing function is applied.

        Args:
            f: A callable that preprocesses the arguments of the wrapped function. Note that `f` must return a _tuple_ of arguments, not a bare argument.
            info: An optional string providing additional information about the `contramap` operation.

        Returns:
            A [`genjax.GenerativeFunction`][] that acts like `self` with a pre-processing function to its arguments.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="contramap"
            import jax, genjax


            # Define a pre-processing function.
            # Note that this function must return a tuple of arguments!
            def add_one(x):
                return (x + 1,)


            @genjax.gen
            def model(x):
                return genjax.normal(x, 1.0) @ "z"


            contramap_model = model.contramap(add_one, info="Add one to input")

            # Use the contramap model
            key = jax.random.PRNGKey(0)
            trace = contramap_model.simulate(key, (2.0,))

            print(trace.render_html())
            ```
        """
        import genjax

        return genjax.contramap(f=f, info=info)(self)

    #####################
    # GenSP / inference #
    #####################

    def marginal(
        self,
        /,
        *,
        selection: Any | None = None,
        algorithm: Any | None = None,
    ) -> "genjax.Marginal[R]":
        from genjax import Selection, marginal

        if selection is None:
            selection = Selection.all()

        return marginal(selection=selection, algorithm=algorithm)(self)


# NOTE: Setup a global handler stack for the `trace` callee sugar.
# C.f. above.
# This stack will not interact with JAX tracers at all
# so it's safe, and will be resolved at JAX tracing time.
GLOBAL_TRACE_OP_HANDLER_STACK: list[Callable[..., Any]] = []


def handle_off_trace_stack(addr, gen_fn: GenerativeFunction[R], args) -> R:
    if GLOBAL_TRACE_OP_HANDLER_STACK:
        handler = GLOBAL_TRACE_OP_HANDLER_STACK[-1]
        return handler(addr, gen_fn, args)
    else:
        raise Exception(
            "Attempting to invoke trace outside of a tracing context.\nIf you want to invoke the generative function closure, and recieve a return value,\ninvoke it with a key."
        )


def push_trace_overload_stack(handler, fn):
    def wrapped(*args):
        GLOBAL_TRACE_OP_HANDLER_STACK.append(handler)
        ret = fn(*args)
        GLOBAL_TRACE_OP_HANDLER_STACK.pop()
        return ret

    return wrapped


@Pytree.dataclass
class IgnoreKwargs(Generic[R], GenerativeFunction[R]):
    wrapped: GenerativeFunction[R]

    def handle_kwargs(self) -> "GenerativeFunction[R]":
        raise NotImplementedError

    def simulate(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Trace[R]:
        (args, _kwargs) = args
        return self.wrapped.simulate(key, args)

    def assess(
        self,
        sample: ChoiceMap,
        args: Arguments,
    ) -> tuple[Score, R]:
        (args, _kwargs) = args
        return self.wrapped.assess(sample, args)

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Arguments,
    ) -> tuple[Trace[Any], Weight]:
        raise NotImplementedError

    def project(
        self,
        key: PRNGKey,
        trace: Trace[Any],
        projection: Projection[Any],
    ) -> Weight:
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        raise NotImplementedError


@Pytree.dataclass
class GenerativeFunctionClosure(Generic[R], GenerativeFunction[R]):
    gen_fn: GenerativeFunction[R]
    args: tuple[Any, ...]
    kwargs: dict[Any, Any]

    def get_gen_fn_with_kwargs(self):
        return self.gen_fn.handle_kwargs()

    # NOTE: Supports callee syntax, and the ability to overload it in callers.
    def __matmul__(self, addr) -> R:
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return handle_off_trace_stack(
                addr,
                maybe_kwarged_gen_fn,
                (self.args, self.kwargs),
            )
        else:
            return handle_off_trace_stack(
                addr,
                self.gen_fn,
                self.args,
            )

    # This override returns `R`, while the superclass returns a `GenerativeFunctionClosure`; this is
    # a hint that subclassing may not be the right relationship here.
    def __call__(self, key: PRNGKey, *args) -> R:  # pyright: ignore[reportIncompatibleMethodOverride]
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key, (*full_args, self.kwargs)
            ).get_retval()
        else:
            return self.gen_fn.simulate(key, full_args).get_retval()

    def __abstract_call__(self, *args) -> R:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.__abstract_call__(*full_args, **self.kwargs)
        else:
            return self.gen_fn.__abstract_call__(*full_args)

    #############################################
    # Support the interface with reduced syntax #
    #############################################

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> Trace[R]:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.simulate(key, full_args)

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Arguments,
    ) -> tuple[Trace[Any], Weight]:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.generate(
                key,
                constraint,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.generate(key, constraint, full_args)

    def project(
        self,
        key: PRNGKey,
        trace: Trace[Any],
        projection: Projection[Any],
    ):
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        raise NotImplementedError

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.assess(
                sample,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.assess(sample, full_args)


@Pytree.dataclass(match_args=True)
class Update(EditRequest):
    constraint: ChoiceMapConstraint

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        gen_fn = tr.get_gen_fn()
        return gen_fn.edit(key, tr, self, argdiffs)
