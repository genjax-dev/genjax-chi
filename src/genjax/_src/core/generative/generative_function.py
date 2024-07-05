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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from penzai.core import formatting_util

from genjax._src.core.generative.choice_map import (
    ChoiceMap,
    ChoiceMapConstraint,
    SelectionProjectRequest,
)
from genjax._src.core.generative.core import (
    Argdiffs,
    Arguments,
    Constraint,
    ConstraintUpdateRequest,
    ImportanceUpdateRequest,
    IncrementalUpdateRequest,
    Retdiff,
    Retval,
    Sample,
    Score,
    UpdateRequest,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import get_trace_shape
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import gfi_boundary, register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    Dict,
    InAxes,
    Int,
    List,
    Optional,
    PRNGKey,
    String,
    Tuple,
    dispatch,
    overload,
    typecheck,
)

register_exclusion(__file__)

#########
# Trace #
#########


class Trace(Pytree):
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
    def get_retval(self) -> Retval:
        """Returns the [`Retval`][genjax.core.Retval] from the [`GenerativeFunction`][genjax.core.GenerativeFunction] invocation which created the [`Trace`][genjax.core.Trace]."""

    @abstractmethod
    def get_score(self) -> Score:
        """Return the [`Score`][genjax.core.Score] of the `Trace`.

        The score must satisfy a particular mathematical specification: it's either an exact density evaluation of $P$ (the distribution over samples) for the sample returned by `Trace.get_sample`, or _a sample from an estimator_ (a density estimate) if the generative function contains _untraced randomness_.

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

    # TODO: deprecated.
    def get_choices(self) -> Sample:
        return self.get_sample()

    @abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction":
        """Returns the [`GenerativeFunction`][genjax.core.GenerativeFunction] whose invocation created the [`Trace`][genjax.core.Trace]."""
        raise NotImplementedError

    @overload
    def update(
        self,
        key: PRNGKey,
        problem: UpdateRequest,
    ) -> Tuple["Trace", Weight, Retdiff, UpdateRequest]:
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, problem)

    @overload
    def update(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        argdiffs: Argdiffs,
    ) -> Tuple["Trace", Weight, Retdiff, UpdateRequest]:
        gen_fn = self.get_gen_fn()
        return gen_fn.update(key, self, choice_map, argdiffs)

    @overload
    def update(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
    ) -> Tuple["Trace", Weight, Retdiff, UpdateRequest]:
        gen_fn = self.get_gen_fn()
        argdiffs = Diff.tree_diff_no_change(self.get_args())
        return gen_fn.update(key, self, choice_map, argdiffs)

    @dispatch
    def update(self, *args) -> Tuple["Trace", Weight, Retdiff, UpdateRequest]:
        """
        This method calls out to the underlying [`GenerativeFunction.update`][genjax.core.GenerativeFunction.update] method - see [`UpdateRequest`][genjax.core.UpdateRequest] and [`update`][genjax.core.GenerativeFunction.update] for more information.
        """
        pass

    @typecheck
    def project(
        self,
        key: PRNGKey,
        request: SelectionProjectRequest,
    ) -> Weight:
        gen_fn = self.get_gen_fn()
        _, w, _, _ = gen_fn.update(
            key,
            self,
            request,
        )
        return -w

    ###################
    # Pretty printing #
    ###################

    def treescope_color(self):
        return self.get_gen_fn().treescope_color()

    ###################
    # Batch semantics #
    ###################

    @property
    def batch_shape(self):
        return len(self.get_score())


@Pytree.dataclass
class EmptyTraceArg(Pytree):
    pass


@Pytree.dataclass
class EmptyTraceRetval(Pytree):
    pass


@Pytree.dataclass
class EmptyTrace(Trace):
    gen_fn: "GenerativeFunction"

    def get_args(self) -> Tuple:
        return (EmptyTraceArg(),)

    def get_retval(self) -> Retval:
        return EmptyTraceRetval()

    def get_score(self) -> Score:
        return 0.0

    def get_sample(self) -> Sample:
        return EmptySample()

    def get_gen_fn(self) -> "GenerativeFunction":
        return self.gen_fn


#######################
# Generative function #
#######################


class GenerativeFunction(Pytree):
    """
    `GenerativeFunction` is the type of _generative functions_, the main computational object in Gen.

    Generative functions are a type of probabilistic program. In terms of their mathematical specification, they come equipped with a few ingredients:

    * (**Distribution over samples**) $P(\\cdot_t, \\cdot_r; a)$ - a probability distribution over samples $t$ and untraced randomness $r$, indexed by arguments $a$. This ingredient is involved in all the interfaces and specifies the distribution over samples which the generative function represents.
    * (**Family of K/L proposals**) $(K(\\cdot_t, \\cdot_{K_r}; u, t), L(\\cdot_t, \\cdot_{L_r}; u, t)) = \\mathcal{F}(u, t)$ - a family of pairs of probabilistic programs (referred to as K and L), indexed by [`UpdateRequest`][genjax.core.UpdateRequest] $u$ and an existing sample $t$. This ingredient supports the [`update`][genjax.core.GenerativeFunction.update] and [`importance`][genjax.core.GenerativeFunction.importance] interface, and is used to specify an SMCP3 move which the generative function must provide in response to an update request. K and L must satisfy additional properties, described further in [`update`][genjax.core.GenerativeFunction.update].
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

    def __call__(self, *args, **kwargs) -> "GenerativeFunctionClosure":
        return GenerativeFunctionClosure(self, args, kwargs)

    def __abstract_call__(self, *args) -> Retval:
        """Used to support abstract evaluation of generative function code.

        Generative functions may customize this to improve compilation time.
        """
        return self.simulate(jax.random.PRNGKey(0), args).get_retval()

    def handle_kwargs(self) -> "GenerativeFunction":
        return IgnoreKwargs(self)

    def get_trace_shape(self, *args) -> Any:
        return get_trace_shape(self, args)

    def get_empty_trace(self, *args) -> Trace:
        data_shape = self.get_trace_shape(*args)
        return jtu.tree_map(lambda v: jnp.zeros(v.shape, dtype=v.dtype), data_shape)

    @classmethod
    def gfi_boundary(cls, c: Callable) -> Callable:
        return gfi_boundary(c)

    @abstractmethod
    def simulate(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Trace:
        """
        Execute the generative function, sampling from its distribution over samples, and return a [`Trace`][genjax.core.Trace].

        ## More on traces

        The [`Trace`][genjax.core.Trace] returned by `simulate` implements its own interface.

        It is responsible for storing the arguments of the invocation ([`Trace.get_args`](core.md#genjax.core.Trace.get_args)), the return value of the generative function ([`Trace.get_retval`](core.md#genjax.core.Trace.get_retval)), the identity of the generative function which produced the trace ([`Trace.get_gen_fn`](core.md#genjax.core.Trace.get_gen_fn)), the sample of traced random choices produced during the invocation ([`Trace.get_sample`](core.md#genjax.core.Trace.get_sample)) and _the score_ of the sample ([`Trace.get_score`](core.md#genjax.core.Trace.get_score)).

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

            Another example, using the same model, composed into [`genjax.repeat`](generative_functions.md#genjax.repeat) - which creates a new generative function, which has the same interface:
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
        raise NotImplementedError

    @overload
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        choice_map: "ChoiceMap",
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        choice_map_constraint = ChoiceMapConstraint(choice_map)
        constraint_request = ConstraintUpdateRequest(choice_map_constraint)
        request = IncrementalUpdateRequest(argdiffs, constraint_request)
        return self.update(key, trace, request)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: UpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        """
        Update a trace in response to an [`UpdateRequest`][genjax.core.UpdateRequest], returning a new [`Trace`][genjax.core.Trace], an incremental [`Weight`][genjax.core.Weight] for the new target, a [`Retdiff`][genjax.core.Retdiff] return value tagged with change information, and a backward [`UpdateRequest`][genjax.core.UpdateRequest] which requests the reverse move (to go back to the original trace).

        The specification of this interface is parametric over the kind of `UpdateRequest` -- responding to an `UpdateRequest` instance requires that the generative function provides an implementation of a sequential Monte Carlo move in the [SMCP3](https://proceedings.mlr.press/v206/lew23a.html) framework. Users of inference algorithms are not expected to understand the ingredients, but inference algorithm developers are.

        Examples:
            Updating a trace in response to a request for a [`Target`][genjax.inference.Target] change induced by a change to the arguments:
            ```python exec="yes" source="material-block" session="core"
            from genjax import gen
            from genjax import normal
            from genjax import EmptyUpdateRequest
            from genjax import Diff
            from genjax import ChoiceMapBuilder as C
            from genjax import UpdateRequestBuilder as U


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
            new_tr, inc_w, retdiff, bwd_prob = model.update(
                key,
                initial_tr,
                U.g(
                    Diff.unknown_change((3.0,)),
                    EmptyUpdateRequest(),
                ),
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

        ## Mathematical ingredients behind update

        The `update` interface exposes [SMCP3 moves](https://proceedings.mlr.press/v206/lew23a.html). Here, we omit the measure theoretic description, and refer interested readers to [the paper](https://proceedings.mlr.press/v206/lew23a.html). Informally, the ingredients of such a move are:

        * The previous target $T$.
        * The new target $T'$.
        * A pair of kernel probabilistic programs, called $K$ and $L$:
            * The K kernel is a kernel probabilistic program which accepts a previous sample $x_{t-1}$ from $T$ as an argument, may sample auxiliary randomness $u_K$, and returns a new sample $x_t$ approximately distributed according to $T'$, along with transformed randomness $u_L$.
            * The L kernel is a kernel probabilistic program which accepts the new sample $x_t$, and provides a density evaluator for the auxiliary randomness $u_L$ which K returns, and an inverter $x_t \\mapsto x_{t-1}$ which is _almost everywhere_ the identity function.

        The specification of these ingredients are encapsulated in the type signature of the `update` interface.

        ## Understanding the `update` interface

        The `update` interface uses the mathematical ingredients described above to perform probability-aware mutations and incremental [`Weight`][genjax.core.Weight] computations on [`Trace`][genjax.core.Trace] instances, which allows Gen to provide automation to support inference agorithms like importance sampling, SMC, MCMC and many more.

        An `UpdateRequest` denotes a function $tr \\mapsto (T, T')$ from traces to a pair of targets (the previous [`Target`][genjax.inference.Target] $T$, and the final [`Target`][genjax.inference.Target] $T'$).

        Several common types of moves can be requested via the `GenericProblem` type:

        ```python exec="yes" source="material-block" session="core"
        from genjax import GenericProblem

        g = GenericProblem(
            Diff.unknown_change((1.0,)),  # "Argdiffs"
            EmptyUpdateRequest(),  # subrequest
        )
        ```

        Creating problem instances is also possible using the `UpdateRequestBuilder`:
        ```python exec="yes" html="true" source="material-block" session="core"
        from genjax import UpdateRequestBuilder as U

        g = U.g(
            Diff.unknown_change((3.0,)),  # "Argdiffs"
            EmptyUpdateRequest(),  # subrequest
        )
        print(g.render_html())
        ```

        `GenericProblem` contains information about changes to the arguments of the generative function ([`Argdiffs`][genjax.core.Argdiffs]) and a subrequest which specifies an additional move to be performed. The subrequest can be a bonafide [`UpdateRequest`][genjax.core.UpdateRequest] itself, or a [`Constraint`][genjax.core.Constraint] (like [`ChoiceMap`][genjax.core.ChoiceMap]).

        ```python exec="yes" html="true" source="material-block" session="core"
        new_tr, inc_w, retdiff, bwd_prob = model.update(
            key,
            initial_tr,
            U.g(Diff.unknown_change((3.0,)), C.kw(v1=3.0)),
        )
        print((new_tr.get_sample()["v1"], w))
        ```

        **Additional notes on [`Argdiffs`][genjax.core.Argdiffs]**

        Argument changes induce changes to the distribution over samples, internal K and L proposals, and (by virtue of changes to $P$) target distributions. The [`Argdiffs`][genjax.core.Argdiffs] type denotes the type of values attached with a _change type_, a piece of data which indicates how the value has changed from the arguments which created the trace. Generative functions can utilize change type information to inform efficient [`update`][genjax.core.GenerativeFunction.update] implementations.
        """
        raise NotImplementedError

    @abstractmethod
    def assess(
        self,
        sample: Sample,
        args: Arguments,
    ) -> Tuple[Score, Retval]:
        """
        Return [the score][genjax.core.Trace.get_score] and [the return value][genjax.core.Trace.get_retval] when the generative function is invoked with the provided arguments, and constrained to take the provided sample as the sampled value.

        It is an error if the provided sample value is off the support of the distribution over the `Sample` type, or otherwise induces a partial constraint on the execution of the generative function (which would require the generative function to provide an `update` implementation which responds to the `UpdateRequest` induced by the [`importance`][genjax.core.GenerativeFunction.importance] interface).

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
        raise NotImplementedError

    @overload
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Arguments,
    ) -> Tuple[Trace, Weight]:
        tr, w, _, _ = self.update(
            key, EmptyTrace(self), ImportanceUpdateRequest(args, constraint)
        )
        return tr, w

    @overload
    def importance(
        self,
        key: PRNGKey,
        constraint: ChoiceMapConstraint,
        args: Arguments,
    ) -> Tuple[Trace, Weight]:
        tr, w, _, _ = self.update(
            key, EmptyTrace(self), ImportanceUpdateRequest(args, constraint)
        )
        return tr, w

    @overload
    def importance(
        self,
        key: PRNGKey,
        constraint: ChoiceMap,
        args: Arguments,
    ) -> Tuple[Trace, Weight]:
        return self.importance(key, ChoiceMapConstraint(constraint), args)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Arguments,
    ) -> Tuple[Trace, Weight]:
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

        Under the hood, creates an [`UpdateRequest`][genjax.core.UpdateRequest] which requests that the generative function respond with a move from the _empty_ trace (the only possible value for _empty_ target $\\delta_\\emptyset$) to the target induced by the generative function for constraint $C$ with arguments $a$.
        """
        raise NotImplementedError

    @typecheck
    def propose(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Tuple[Sample, Score, Retval]:
        """
        Samples a [`Sample`][genjax.core.Sample] and any untraced randomness $r$ from the generative function's distribution over samples ($P$), and returns the [`Score`][genjax.core.Score] of that sample under the distribution, and the [`Retval`][genjax.core.Retval] of the generative function's return value function $f(r, t, a)$ for the sample and untraced randomness.
        """
        tr = self.simulate(key, args)
        sample = tr.get_sample()
        score = tr.get_score()
        retval = tr.get_retval()
        return sample, score, retval

    # NOTE: Supports pretty printing in penzai.
    def treescope_color(self):
        type_string = str(type(self))
        return formatting_util.color_from_string(type_string)

    ######################################################
    # Convenience: postfix syntax for combinators / DSLs #
    ######################################################

    ###############
    # Combinators #
    ###############

    def vmap(self, /, *, in_axes: InAxes = 0) -> "GenerativeFunction":
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

    def repeat(self, /, *, n: Int) -> "GenerativeFunction":
        """
        Returns a [`GenerativeFunction`][genjax.GenerativeFunction] that samples from `self` `n` times, returning a vector of `n` results and nesting traced values under an index.

        This combinator is useful for creating multiple samples from the same generative model in a batched manner.

        Args:
            n: The number of times to sample from the generative function.

        Returns:
            A new [`GenerativeFunction`][genjax.GenerativeFunction] that samples from the original function `n` times.
        """
        import genjax

        return genjax.repeat(n=n)(self)

    def scan(self, /, *, n: Int) -> "GenerativeFunction":
        import genjax

        return genjax.scan(n=n)(self)

    def mask(self) -> "GenerativeFunction":
        import genjax

        return genjax.mask(self)

    def or_else(self, gen_fn: "GenerativeFunction") -> "GenerativeFunction":
        """
        Returns a [`GenerativeFunction`][genjax.GenerativeFunction] that accepts

        - a boolean argument
        - an argument tuple for `self`
        - an argument tuple for the supplied `gen_fn`

        and acts like `self` when the boolean is `True` or like `gen_fn` otherwise.

        Args:
            gen_fn: called when the boolean argument is `False`.

        Returns:
            [`GenerativeFunction`][genjax.GenerativeFunction]

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

        return genjax.or_else(gen_fn)(self)

    def map_addresses(self, /, *, mapping: dict) -> "GenerativeFunction":
        import genjax

        return genjax.map_addresses(mapping=mapping)(self)

    def switch(self, *branches: "GenerativeFunction") -> "GenerativeFunction":
        import genjax

        return genjax.switch(*branches)(self)

    def mix(self, *fns: "GenerativeFunction") -> "GenerativeFunction":
        import genjax

        return genjax.mix(*fns)(self)

    def dimap(
        self,
        /,
        *,
        pre: Callable,
        post: Callable,
        info: Optional[String] = None,
    ) -> "GenerativeFunction":
        import genjax

        return genjax.dimap(pre=pre, post=post, info=info)(self)

    def map(
        self, f: Callable, *, info: Optional[String] = None
    ) -> "GenerativeFunction":
        import genjax

        return genjax.map(f=f, info=info)

    def contramap(
        self, f: Callable, *, info: Optional[String] = None
    ) -> "GenerativeFunction":
        import genjax

        return genjax.contramap(f=f, info=info)

    #####################
    # GenSP / inference #
    #####################

    def marginal(
        self,
        /,
        *,
        selection: Optional[Any] = None,
        algorithm: Optional[Any] = None,
    ) -> "GenerativeFunction":
        from genjax import Selection, marginal

        if selection is None:
            selection = Selection.all()

        return marginal(selection=selection, algorithm=algorithm)(self)

    def target(
        self,
        /,
        *,
        constraint: Constraint,
        args: Tuple,
    ):
        from genjax import Target

        return Target(
            self,
            args,
            constraint,
        )


# NOTE: Setup a global handler stack for the `trace` callee sugar.
# C.f. above.
# This stack will not interact with JAX tracers at all
# so it's safe, and will be resolved at JAX tracing time.
GLOBAL_TRACE_OP_HANDLER_STACK: List[Callable] = []


def handle_off_trace_stack(addr, gen_fn: GenerativeFunction, args):
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
class IgnoreKwargs(GenerativeFunction):
    wrapped: "GenerativeFunction"

    def handle_kwargs(self) -> "GenerativeFunction":
        raise NotImplementedError

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Trace:
        (args, _kwargs) = args
        return self.wrapped.simulate(key, args)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: UpdateRequest,
        argdiffs: Argdiffs,
    ):
        (argdiffs, _kwargdiffs) = argdiffs
        return self.wrapped.update(key, trace, update_request, argdiffs)


@Pytree.dataclass(match_args=True)
class Target(Pytree):
    """
    A `Target` represents an unnormalized target distribution induced by conditioning a generative function on a [`Constraint`](core.md#genjax.core.Constraint).

    Targets are created by providing a generative function, arguments to the generative function, and a constraint.

    Examples:
        Creating a target from a generative function, by providing arguments and a constraint:
        ```python exec="yes" html="true" source="material-block" session="core"
        import genjax
        from genjax import ChoiceMapBuilder as C
        from genjax.inference import Target


        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 1.0) @ "y"
            return x


        target = Target(model, (), C["y"].set(3.0))
        print(target.render_html())
        ```
    """

    p: GenerativeFunction
    args: Arguments
    constraint: Constraint


@Pytree.dataclass
class GenerativeFunctionClosure(GenerativeFunction):
    gen_fn: GenerativeFunction
    args: Tuple
    kwargs: Dict

    def get_gen_fn_with_kwargs(self):
        return self.gen_fn.handle_kwargs()

    # NOTE: Supports callee syntax, and the ability to overload it in callers.
    def __matmul__(self, addr):
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

    def __call__(self, key: PRNGKey, *args) -> Any:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key, (*full_args, self.kwargs)
            ).get_retval()
        else:
            return self.gen_fn.simulate(key, full_args).get_retval()

    def __abstract_call__(self, *args) -> Any:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.__abstract_call__(*full_args, **self.kwargs)
        else:
            return self.gen_fn.__abstract_call__(*full_args)

    #############################################
    # Support the interface with reduced syntax #
    #############################################

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.simulate(
                key,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.simulate(key, full_args)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        problem: UpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        match problem:
            case IncrementalUpdateRequest(argdiffs, subrequest):
                full_argdiffs = (*self.args, *argdiffs)
                if self.kwargs:
                    maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
                    return maybe_kwarged_gen_fn.update(
                        key,
                        trace,
                        IncrementalUpdateRequest(
                            (full_argdiffs, self.kwargs),
                            subrequest,
                        ),
                    )
            case _:
                raise NotImplementedError

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Retval]:
        full_args = (*self.args, *args)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.assess(
                sample,
                (full_args, self.kwargs),
            )
        else:
            return self.gen_fn.assess(sample, full_args)
