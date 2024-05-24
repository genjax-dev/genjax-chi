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

from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import get_trace_shape, staged_and
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import gfi_boundary, register_exclusion
from genjax._src.core.typing import (
    Annotated,
    Any,
    Bool,
    BoolArray,
    Callable,
    Dict,
    FloatArray,
    InAxes,
    Int,
    IntArray,
    Is,
    List,
    Optional,
    PRNGKey,
    Tuple,
    static_check_is_concrete,
    typecheck,
)

register_exclusion(__file__)

#####################################
# Special generative function types #
#####################################

Weight = Annotated[
    float | FloatArray,
    Is[lambda arr: jnp.array(arr, copy=False).shape == ()],
]
"""
A _weight_ is a density ratio (an importance weight), whose mathematical content is described in [`update`][genjax.core.GenerativeFunction.update].

The type `Weight` does not enforce any meaningful mathematical invariants, but is used to denote the type of weights in GenJAX, to improve readability and parsing of interface specifications / expectations.
"""
Score = Annotated[
    float | FloatArray,
    Is[lambda arr: jnp.array(arr, copy=False).shape == ()],
]
"""
A _score_ is a specific density ratio, described fully in [`simulate`][genjax.core.GenerativeFunction.simulate].

The type `Score` does not enforce any meaningful mathematical invariants, but is used to denote the type of scores in the GenJAX system, to improve readability and parsing of interface specifications.

Under type checking, the type `Score` enforces that the value must be a scalar floating point number.
"""

Arguments = Tuple
"""
`Arguments` is the type of argument values to generative functions. It is a type alias for `Tuple`, and is used to improve readability and parsing of interface specifications.
"""

Retval = Any
"""
`Retval` is the type of return values from the return value function of a generative function. It is a type alias for `Any`, and is used to improve readability and parsing of interface specifications.
"""

Argdiffs = Annotated[
    Tuple,
    Is[lambda v: Diff.static_check_tree_diff(v)],
]
"""
`Argdiffs` is the type of argument values with an attached `ChangeType` (c.f. [`update`][genjax.core.GenerativeFunction.update]).

When used under type checking, `Retdiff` assumes that the argument values are `Pytree` (either, defined via GenJAX's `Pytree` interface or registered with JAX's system). For each argument, it checks that _the leaves_ are `Diff` type with attached `ChangeType`.
"""


Retdiff = Annotated[
    Retval,
    Is[lambda v: Diff.static_check_tree_diff(v)],
]
"""
`Retdiff` is the type of return values with an attached `ChangeType` (c.f. [`update`][genjax.core.GenerativeFunction.update]).

When used under type checking, `Retdiff` assumes that the return value is a `Pytree` (either, defined via GenJAX's `Pytree` interface or registered with JAX's system). It checks that _the leaves_ are `Diff` type with attached `ChangeType`.
"""


#########################
# Update specifications #
#########################


class UpdateProblem(Pytree):
    """
    An `UpdateProblem` is a request to update a trace of a generative function. Generative functions respond to `UpdateProblem` instances by providing an [`update`][genjax.core.GenerativeFunction.update] implementation.

    Updating a trace is a common operation in inference processes, but naively mutating the trace will invalidate the mathematical invariants that Gen retains. `UpdateProblem` instances denote requests for _SMC moves_ in the framework of [SMCP3](https://proceedings.mlr.press/v206/lew23a.html), which preserve these invariants.

    """

    @classmethod
    def empty(cls):
        return EmptyProblem()

    @classmethod
    def maybe(cls, flag: Bool | BoolArray, problem: "UpdateProblem"):
        return MaskedProblem.maybe_empty(flag, problem)


@Pytree.dataclass
class EmptyProblem(UpdateProblem):
    pass


@Pytree.dataclass(match_args=True)
class MaskedProblem(UpdateProblem):
    flag: Bool | BoolArray
    problem: UpdateProblem

    @classmethod
    def maybe_empty(cls, f: BoolArray, problem: UpdateProblem):
        match problem:
            case MaskedProblem(flag, subproblem):
                return MaskedProblem(staged_and(f, flag), subproblem)
            case _:
                static_bool_check = static_check_is_concrete(f) and isinstance(f, Bool)
                return (
                    problem
                    if static_bool_check and f
                    else EmptyProblem()
                    if static_bool_check
                    else MaskedProblem(f, problem)
                )


@Pytree.dataclass
class SumProblem(UpdateProblem):
    idx: Int | IntArray
    problems: List[UpdateProblem]


@Pytree.dataclass(match_args=True)
class ImportanceProblem(UpdateProblem):
    constraint: "Constraint"


@Pytree.dataclass
class ProjectProblem(UpdateProblem):
    pass


###############
# Constraints #
###############


class Constraint(UpdateProblem):
    """
    An `Constraint` is a type of `UpdateProblem` specified by a function from the [`Sample`][genjax.core.Sample] space of the generative function to a value space `Y`, and a target value `v` in `Y`. In other words, the tuple $(S \\mapsto Y, v \\in Y)$.

    Just like all `UpdateProblem` instances, the generative function must respond to the request to update a trace to satisfy the constraint by providing an [`update`][genjax.core.GenerativeFunction.update] implementation which implements an SMCP3 move that transforms the provided trace to satisfy the specification.

    Constraints can also be used to construct [`ImportanceProblem`](genjax.core.ImportanceProblem) instances, which are used to implement the [`importance`][genjax.core.GenerativeFunction.importance] interface. This interface implements a restricted SMCP3 move, from the empty target, to the target induced by the constraint.
    """


@Pytree.dataclass
class EmptyConstraint(Constraint):
    """
    An `EmptyConstraint` encodes the lack of a constraint.

    Formally, `EmptyConstraint(x)` represents the constraint `(x $\\mapsto$ (), ())`.
    """

    pass


@Pytree.dataclass
class EqualityConstraint(Constraint):
    """
    An `EqualityConstraint` encodes the constraint that the value output by a
    distribution is equal to a provided value.

    Formally, `EqualityConstraint(x)` represents the constraint `(x $\\mapsto$ x, x)`.
    """

    x: Any


@Pytree.dataclass(match_args=True)
class MaskedConstraint(Constraint):
    """
    A `MaskedConstraint` encodes a possible constraint.

    Formally, `MaskedConstraint(f: Bool, c: Constraint)` represents the constraint `Option((x $\\mapsto$ x, x))`,
    where the None case is represented by `EmptyConstraint`.
    """

    flag: Bool | BoolArray
    constraint: Constraint


@Pytree.dataclass
class SumConstraint(Constraint):
    """
    A `SumConstraint` encodes that one of a set of possible constraints is active _at runtime_, using a provided index.

    Formally, `SumConstraint(idx: IntArray, cs: List[Constraint])` represents the constraint (`x` $\\mapsto$ `xs[idx]`, `ys[idx]`).
    """

    idx: IntArray
    constraint: List[Constraint]


@Pytree.dataclass
class IntervalConstraint(Constraint):
    """
    An IntervalConstraint encodes the constraint that the value output by a
    distribution on the reals lies within a given interval.

    Formally, `IntervalConstraint(a, b)` represents the constraint (`x` $\\mapsto$ `a` $\\leq$ `x` $\\leq$ `b`, `True`).
    """

    a: FloatArray
    b: FloatArray


@Pytree.dataclass
class BijectiveConstraint(Constraint):
    """
    A `BijectiveConstraint` encodes the constraint that the value output by a distribution
    must, under a bijective transformation, be equal to the value provided to the constraint.

    Formally, `BijectiveConstraint(bwd, v)` represents the constraint `(x $\\mapsto$ inverse(bwd)(x), v)`.
    """

    bwd: Callable[[Any], "Sample"]
    v: Any


###########
# Samples #
###########


class Sample(Pytree):
    """A `Sample` is a value which can be sampled from generative functions. Samples can be scalar values, or map-like values ([`ChoiceMap`][genjax.core.ChoiceMap]). Different sample types can induce different interfaces: `ChoiceMap`, for instance, supports interfaces for accessing sub-maps and values."""


@Pytree.dataclass
class EmptySample(Sample):
    pass


@Pytree.dataclass(match_args=True)
class MaskedSample(Sample):
    flag: Bool | BoolArray
    sample: Sample


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

    def update(
        self,
        key: PRNGKey,
        problem: UpdateProblem,
        argdiffs: Optional[Argdiffs] = None,
    ) -> Tuple["Trace", Weight, Retdiff, UpdateProblem]:
        """
        This method calls out to the underlying [`GenerativeFunction.update`][genjax.core.GenerativeFunction.update] method - see [`UpdateProblem`][genjax.core.UpdateProblem] and [`update`][genjax.core.GenerativeFunction.update] for more information.
        """
        gen_fn = self.get_gen_fn()
        if argdiffs:
            check = Diff.static_check_tree_diff(argdiffs)
            argdiffs = argdiffs if check else Diff.tree_diff_unknown_change(argdiffs)
            return gen_fn.update(key, self, problem, argdiffs)
        else:
            old_args = self.get_args()
            argdiffs = Diff.tree_diff_no_change(old_args)
            return gen_fn.update(key, self, problem, argdiffs)

    @typecheck
    def project(
        self,
        key: PRNGKey,
        problem: ProjectProblem,
    ) -> Weight:
        gen_fn = self.get_gen_fn()
        _, w, _, _ = gen_fn.update(key, self, problem, Diff.no_change(self.get_args()))
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

    Generative functions are a type of probabilistic program. In terms of their specification, they come equipped with a few mathematical ingredients:

    * (**Distribution over samples**) $P(\\cdot_t, \\cdot_r; a)$ - a probability distribution over samples $t$ and untraced randomness $r$, indexed by arguments $a$. This ingredient supports the [`simulate`][genjax.core.GenerativeFunction.simulate] and [`assess`][genjax.core.GenerativeFunction.assess] interfaces, and specifies the distribution over samples which the generative function represents.
    * (**Family of K/L proposals**) $(K(\\cdot_t, \\cdot_{K_r}; u, t), L(\\cdot_t, \\cdot_{L_r}; u, t)) = \\mathcal{F}(u, t)$ - a family of pairs of probabilistic programs (referred to as K and L), indexed by [`UpdateProblem`][genjax.core.UpdateProblem] $u$ and an existing sample $t$. This ingredient supports the [`update`][genjax.core.GenerativeFunction.update] and [`importance`][genjax.core.GenerativeFunction.importance] interface, and is used to specify an SMCP3 move which the generative function must provide in response to an update request. K and L must satisfy additional properties, described further in [`update`][genjax.core.GenerativeFunction.update].
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
        """Used to support JAX tracing, although this default implementation involves no
        JAX operations (it takes a fixed-key sample from the return value).

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
        Execute the generative function, sampling from its distribution over samples, and return a [`Trace`](core.md#genjax.core.Trace).

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

            Another example, using the same model, composed into [`genjax.repeat_combinator`](generative_functions.md#genjax.repeat_combinator) - which creates a new generative function, which has the same interface:
            ```python exec="yes" html="true" source="material-block" session="core"
            @genjax.gen
            def model():
                x = genjax.normal(0.0, 1.0) @ "x"
                return x


            key = PRNGKey(0)
            tr = model.repeat(num_repeats=10).simulate(key, ())
            print(tr.render_html())
            ```

            (**Fun, flirty, fast ... parallel?**) Feel free to use `jax.jit` and `jax.vmap`!
            ```python exec="yes" html="true" source="material-block" session="core"
            key = PRNGKey(0)
            sub_keys = split(key, 10)
            sim = model.repeat(num_repeats=10).simulate
            tr = jit(vmap(sim, in_axes=(0, None)))(sub_keys, ())
            print(tr.render_html())
            ```


        The trace returned by `simulate` has the arguments of the invocation ([`Trace.get_args`](core.md#genjax.core.Trace.get_args)), the return value of the generative function ([`Trace.get_retval`](core.md#genjax.core.Trace.get_retval)), the identity of the generative function which produced the trace ([`Trace.get_gen_fn`](core.md#genjax.core.Trace.get_gen_fn)), the sample of traced random choices produced during the invocation ([`Trace.get_sample`](core.md#genjax.core.Trace.get_sample)) and _the score_ of the sample ([`Trace.get_score`](core.md#genjax.core.Trace.get_score)).

        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        """
        Update a trace in response to an [`UpdateProblem`][genjax.core.UpdateProblem], returning a new [`Trace`][genjax.core.Trace], a proper [`Weight`][genjax.core.Weight] for the new target, a [`Retdiff`][genjax.core.Retdiff] return value tagged with change information, and a backward [`UpdateProblem`][genjax.core.UpdateProblem] which requests the reverse move (to go back to the original trace).

        The specification of this interface is parametric over the kind of `UpdateProblem` -- responding to an `UpdateProblem` instance requires that the generative function provides an implementation of a sequential Monte Carlo move in the [SMCP3](https://proceedings.mlr.press/v206/lew23a.html) framework.

        **Mathematical specification of an SMCP3 move**

        Here, we omit the measure theoretic formalization, and refer interested readers to [the paper](https://proceedings.mlr.press/v206/lew23a.html). The ingredients of such a move are:

        * The previous target $T$.
        * The new target $T'$.
        * A pair of kernel probabilistic programs, called $K$ and $L$:
            * The K kernel is a kernel probabilistic program which accepts a previous sample $x_{t-1}$ from $T$ as an argument, may sample auxiliary randomness $u_K$, and returns a new sample $x_t$ approximately distributed according to $T'$, along with transformed randomness $u_L$.
            * The L kernel is a kernel probabilistic program which accepts the new sample $x_t$, and provides a density evaluator for the auxiliary randomness $u_L$ which K returns, and an inverter $x_t \\mapsto x_{t-1}$ which is _almost everywhere_ the identity function.

        These ingredients are encapsulated in the types of the `update` interface:

        **What are `Argdiffs`?**

        The `Argdiffs` type denotes the type of values attached with a _change type_, a piece of data which indicates how the value has changed from the arguments which created the trace.

        Changing the value of the arguments is part of specifying the previous and new targets in the update request: `Argdiffs` are a way to inform Gen about specific changes to the arguments as part of the `update` request, and (when combined with `UpdateProblem`) can be used to support update optimizations.

        **Specifying an SMCP3 move via `UpdateProblem` and `Argdiffs`**

        An `UpdateProblem`, along with the new arguments (the _primals_ of `Argdiffs` -- primals meaning the values without change type information) denotes a function $Tr \\rightarrow (T, T')$ from the type $Tr$ of traces to a pair of targets (the previous target $T$, and the final target $T'$).

        The generative function is responsible for providing an [`update`][genjax.core.GenerativeFunction.update] implementation which responds to the request, by implementing an SMCP3 move which satisfies the specification.

        **Common types of `UpdateProblem`**

        Constraints are a simple type of `UpdateProblem` which specify a move from ... A common type of constraint / problem is [`ChoiceMap`][genjax.core.ChoiceMap].
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

        It is an error if the provided sample value is off the support of the distribution over the `Sample` type, or otherwise induces a partial constraint on the execution of the generative function (which would require the generative function to provide an `update` implementation which responds to the `UpdateProblem` induced by the [`importance`][genjax.core.GenerativeFunction.importance] interface).

        This method is similar to density evaluation interfaces for distributions.
        """
        raise NotImplementedError

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Arguments,
    ) -> Tuple[Trace, Weight]:
        """
        Returns a properly weighted pair, a [`Trace`][genjax.core.Trace] and a [`Weight`][genjax.core.Weight], properly weighted for the target induced by the generative function for the provided constraint and arguments.

        Formally, creates an `UpdateProblem` which requests that the generative function respond with a move from the _empty_ trace (the only possible value for _empty_ target $\\delta_\\emptyset$) to the target induced by the generative function for constraint $C$ with arguments $a$.
        """
        importance_problem = ImportanceProblem(constraint)
        tr, w, _, _ = self.update(
            key,
            EmptyTrace(self),
            importance_problem,
            Diff.unknown_change(args),
        )
        return tr, w

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

    def vmap(
        self,
        *args,
        in_axes: InAxes = 0,
    ) -> "GenerativeFunction":
        from genjax import vmap_combinator

        return (
            vmap_combinator(self, in_axes=in_axes)(*args)
            if args
            else vmap_combinator(self, in_axes=in_axes)
        )

    def repeat(
        self,
        *args,
        num_repeats: Int,
    ) -> "GenerativeFunction":
        from genjax import repeat_combinator

        return (
            repeat_combinator(self, num_repeats=num_repeats)(*args)
            if args
            else repeat_combinator(self, num_repeats=num_repeats)
        )

    def scan(
        self,
        *args,
        max_length: Int,
    ) -> "GenerativeFunction":
        from genjax import scan_combinator

        return (
            scan_combinator(self, max_length=max_length)(*args)
            if args
            else scan_combinator(self, max_length=max_length)
        )

    def mask(
        self,
        *args,
    ) -> "GenerativeFunction":
        from genjax import mask_combinator

        return mask_combinator(self)(*args) if args else mask_combinator(self)

    def or_else(
        self,
        gen_fn: "GenerativeFunction",
        *args,
    ) -> "GenerativeFunction":
        from genjax import cond_combinator

        return (
            cond_combinator(self, gen_fn)(*args)
            if args
            else cond_combinator(self, gen_fn)
        )

    def addr_bij(
        self,
        address_bijection: dict,
        *args,
    ) -> "GenerativeFunction":
        from genjax import address_bijection_combinator

        return (
            address_bijection_combinator(self, address_bijection=address_bijection)(
                *args
            )
            if args
            else address_bijection_combinator(self, address_bijection=address_bijection)
        )

    def switch(
        self,
        branches: List["GenerativeFunction"],
        *args,
    ) -> "GenerativeFunction":
        from genjax import switch_combinator

        return (
            switch_combinator(self, *branches)(*args)
            if args
            else switch_combinator(self, *branches)
        )

    def mix(
        self,
        gen_fn: "GenerativeFunction",
        *args,
    ) -> "GenerativeFunction":
        from genjax import mixture_combinator

        return (
            mixture_combinator(self, gen_fn)(*args)
            if args
            else mixture_combinator(self, gen_fn)
        )

    def attach(
        self,
        *args,
        **kwargs,
    ) -> "GenerativeFunction":
        from genjax.inference.smc import attach_combinator

        return (
            attach_combinator(self, **kwargs)(*args)
            if args
            else attach_combinator(self, **kwargs)
        )

    #####################
    # GenSP / inference #
    #####################

    def marginal(
        self,
        *args,
        select_or_addr: Optional[Any] = None,
        algorithm: Optional[Any] = None,
    ) -> "GenerativeFunction":
        from genjax import marginal

        return (
            marginal(self, select_or_addr=select_or_addr, algorithm=algorithm)(*args)
            if args
            else marginal(self, select_or_addr=select_or_addr, algorithm=algorithm)
        )

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
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ):
        (argdiffs, _kwargdiffs) = argdiffs
        return self.wrapped.update(key, trace, update_problem, argdiffs)


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
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        full_argdiffs = (*self.args, *argdiffs)
        if self.kwargs:
            maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
            return maybe_kwarged_gen_fn.update(
                key,
                trace,
                problem,
                (full_argdiffs, self.kwargs),
            )
        else:
            return self.gen_fn.update(key, trace, problem, full_argdiffs)

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
