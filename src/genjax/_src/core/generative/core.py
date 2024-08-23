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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from penzai.core import formatting_util

from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import Flag, get_trace_shape
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import gfi_boundary
from genjax._src.core.typing import (
    Annotated,
    Any,
    Callable,
    FloatArray,
    Generic,
    InAxes,
    Int,
    IntArray,
    Is,
    PRNGKey,
    String,
    TypeVar,
    typecheck,
)

# Import `genjax` so static typecheckers can see the circular reference to "genjax.ChoiceMap" below.
if TYPE_CHECKING:
    import genjax

_C = TypeVar("_C", bound=Callable[..., Any])
ArgTuple = TypeVar("ArgTuple", bound=tuple)

# Generative Function type variables
R = TypeVar("R")
"""
Generic denoting the return type of a generative function.
"""

S = TypeVar("S")

Carry = TypeVar("Carry")
Y = TypeVar("Y")

#####################################
# Special generative function types #
#####################################

Weight = FloatArray
"""
A _weight_ is a density ratio which often occurs in the context of proper weighting for [`Target`][genjax.inference.Target] distributions, or in Gen's [`update`][genjax.core.GenerativeFunction.update] interface, whose mathematical content is described in [`update`][genjax.core.GenerativeFunction.update].

The type `Weight` does not enforce any meaningful mathematical invariants, but is used to denote the type of weights in GenJAX, to improve readability and parsing of interface specifications / expectations.
"""
Score = FloatArray
"""
A _score_ is a density ratio, described fully in [`simulate`][genjax.core.GenerativeFunction.simulate].

The type `Score` does not enforce any meaningful mathematical invariants, but is used to denote the type of scores in the GenJAX system, to improve readability and parsing of interface specifications.
"""

Arguments = tuple
"""
`Arguments` is the type of argument values to generative functions. It is a type alias for `Tuple`, and is used to improve readability and parsing of interface specifications.
"""

Argdiffs = Annotated[
    tuple,
    Is[Diff.static_check_tree_diff],
]
"""
`Argdiffs` is the type of argument values with an attached `ChangeType` (c.f. [`update`][genjax.core.GenerativeFunction.update]).

When used under type checking, `Retdiff` assumes that the argument values are `Pytree` (either, defined via GenJAX's `Pytree` interface or registered with JAX's system). For each argument, it checks that _the leaves_ are `Diff` type with attached `ChangeType`.
"""


Retdiff = Annotated[
    R,
    Is[Diff.static_check_tree_diff],
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
    An `UpdateProblem` is a request to update a trace of a generative function. Generative functions respond to instances of subtypes of `UpdateProblem` by providing an [`update`][genjax.core.GenerativeFunction.update] implementation.

    Updating a trace is a common operation in inference processes, but naively mutating the trace will invalidate the mathematical invariants that Gen retains. `UpdateProblem` instances denote requests for _SMC moves_ in the framework of [SMCP3](https://proceedings.mlr.press/v206/lew23a.html), which preserve these invariants.
    """


@Pytree.dataclass
class EmptyProblem(UpdateProblem):
    pass


@Pytree.dataclass(match_args=True)
class MaskedProblem(UpdateProblem):
    flag: Flag
    problem: UpdateProblem

    @classmethod
    def maybe_empty(cls, f: Flag, problem: UpdateProblem):
        match problem:
            case MaskedProblem(flag, subproblem):
                return MaskedProblem(f.and_(flag), subproblem)
            case _:
                return (
                    problem
                    if f.concrete_true()
                    else EmptyProblem()
                    if f.concrete_false()
                    else MaskedProblem(f, problem)
                )


@Pytree.dataclass
class SumProblem(UpdateProblem):
    idx: Int | IntArray
    problems: list[UpdateProblem]


@Pytree.dataclass(match_args=True)
class GenericProblem(UpdateProblem):
    argdiffs: Argdiffs
    subproblem: UpdateProblem


@Pytree.dataclass(match_args=True)
class ImportanceProblem(UpdateProblem):
    constraint: "Constraint"


@Pytree.dataclass
class ProjectProblem(UpdateProblem):
    pass


class UpdateProblemBuilder(Pytree):
    @classmethod
    def empty(cls):
        return EmptyProblem()

    @classmethod
    def maybe(cls, flag: Flag, problem: "UpdateProblem"):
        return MaskedProblem.maybe_empty(flag, problem)

    @classmethod
    def g(cls, argdiffs: Argdiffs, subproblem: "UpdateProblem") -> "GenericProblem":
        return GenericProblem(argdiffs, subproblem)


###############
# Constraints #
###############


class Constraint(UpdateProblem):
    """
    `Constraint` is a type of [`UpdateProblem`][genjax.core.UpdateProblem] specified by a function from the [`Sample`][genjax.core.Sample] space of the generative function to a value space `Y`, and a target value `v` in `Y`. In other words, a [`Constraint`][genjax.core.Constraint] denotes the pair $(S \\mapsto Y, v \\in Y)$.

    Constraints represent a request to force a value to satisfy a predicate. Just like all [`UpdateProblem`][genjax.core.UpdateProblem] instances, the generative function must respond to the request to update a trace to satisfy the constraint by providing an [`update`][genjax.core.GenerativeFunction.update] implementation which implements an SMCP3 move that transforms the provided trace to satisfy the specification.

    Constraints can also be used to construct `ImportanceProblem` instances, which are used to implement the [`importance`][genjax.core.GenerativeFunction.importance] interface. This interface implements a restricted SMCP3 move, from the empty target, to the target induced by the constraint.
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

    flag: Flag
    constraint: Constraint


@Pytree.dataclass
class SumConstraint(Constraint):
    """
    A `SumConstraint` encodes that one of a set of possible constraints is active _at runtime_, using a provided index.

    Formally, `SumConstraint(idx: IntArray, cs: List[Constraint])` represents the constraint (`x` $\\mapsto$ `xs[idx]`, `ys[idx]`).
    """

    idx: IntArray
    constraint: list[Constraint]


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
    flag: Flag
    sample: Sample


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

    # TODO: deprecated.
    @typecheck
    def get_choices(self) -> "genjax.ChoiceMap":
        """Version of [`genjax.Trace.get_sample`][] for traces where the sample is an instance of [`genjax.ChoiceMap`][]."""
        return self.get_sample()  # type: ignore

    @abstractmethod
    def get_gen_fn(self) -> "GenerativeFunction[R]":
        """Returns the [`GenerativeFunction`][genjax.core.GenerativeFunction] whose invocation created the [`Trace`][genjax.core.Trace]."""
        raise NotImplementedError

    def update(
        self,
        key: PRNGKey,
        problem: GenericProblem | UpdateProblem,
        argdiffs: tuple | None = None,
    ) -> tuple["Trace[R]", Weight, Retdiff, UpdateProblem]:
        """
        This method calls out to the underlying [`GenerativeFunction.update`][genjax.core.GenerativeFunction.update] method - see [`UpdateProblem`][genjax.core.UpdateProblem] and [`update`][genjax.core.GenerativeFunction.update] for more information.
        """
        if isinstance(problem, GenericProblem) and argdiffs is None:
            return self.get_gen_fn().update(key, self, problem)
        elif isinstance(problem, UpdateProblem):
            return self.get_gen_fn().update(
                key,
                self,
                GenericProblem(Diff.tree_diff_no_change(self.get_args()), problem),
            )
        else:
            raise NotImplementedError(
                "Supply either a GenericProblem or an UpdateProblem, possibly with argdiffs"
            )

    @typecheck
    def project(
        self,
        key: PRNGKey,
        problem: ProjectProblem,
    ) -> Weight:
        gen_fn = self.get_gen_fn()
        _, w, _, _ = gen_fn.update(
            key,
            self,
            GenericProblem(Diff.no_change(self.get_args()), problem),
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


# TODO figure out if / how we can remove this idea, this is masquerading as a trace!
@Pytree.dataclass
class EmptyTrace(Trace[R]):
    gen_fn: "GenerativeFunction[R]"

    def get_args(self) -> tuple[EmptyTraceArg]:
        return (EmptyTraceArg(),)

    def get_retval(self) -> R:
        raise NotImplementedError("Not gonna happen!")

    def get_score(self) -> Score:
        return jnp.array(0.0)

    def get_sample(self) -> Sample:
        return EmptySample()

    def get_gen_fn(self) -> "GenerativeFunction[R]":
        return self.gen_fn


#######################
# Generative function #
#######################


class GenerativeFunction(Generic[R], Pytree):
    """
    `GenerativeFunction` is the type of _generative functions_, the main computational object in Gen.

    Generative functions are a type of probabilistic program. In terms of their mathematical specification, they come equipped with a few ingredients:

    * (**Distribution over samples**) $P(\\cdot_t, \\cdot_r; a)$ - a probability distribution over samples $t$ and untraced randomness $r$, indexed by arguments $a$. This ingredient is involved in all the interfaces and specifies the distribution over samples which the generative function represents.
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

    def __call__(self, *args, **kwargs) -> "GenerativeFunctionClosure[R]":
        return GenerativeFunctionClosure(self, args, kwargs)

    def __abstract_call__(self, *args) -> R:
        """Used to support JAX tracing, although this default implementation involves no
        JAX operations (it takes a fixed-key sample from the return value).

        Generative functions may customize this to improve compilation time.
        """
        return self.simulate(jax.random.PRNGKey(0), args).get_retval()

    def handle_kwargs(self) -> "GenerativeFunction[R]":
        return IgnoreKwargs(self)

    def get_trace_shape(self, *args) -> Any:
        return get_trace_shape(self, args)

    def get_empty_trace(self, *args) -> Trace[R]:
        trace_template = self.get_trace_shape(*args)
        return jtu.tree_map(lambda v: jnp.zeros(v.shape, dtype=v.dtype), trace_template)

    @classmethod
    def gfi_boundary(cls, c: _C) -> _C:
        return gfi_boundary(c)

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
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        key: PRNGKey,
        trace: Trace[R],
        update_problem: GenericProblem,
    ) -> tuple[Trace[R], Weight, Retdiff, UpdateProblem]:
        """
        Update a trace in response to an [`UpdateProblem`][genjax.core.UpdateProblem], returning a new [`Trace`][genjax.core.Trace], an incremental [`Weight`][genjax.core.Weight] for the new target, a [`Retdiff`][genjax.core.Retdiff] return value tagged with change information, and a backward [`UpdateProblem`][genjax.core.UpdateProblem] which requests the reverse move (to go back to the original trace).

        The specification of this interface is parametric over the kind of `UpdateProblem` -- responding to an `UpdateProblem` instance requires that the generative function provides an implementation of a sequential Monte Carlo move in the [SMCP3](https://proceedings.mlr.press/v206/lew23a.html) framework. Users of inference algorithms are not expected to understand the ingredients, but inference algorithm developers are.

        Examples:
            Updating a trace in response to a request for a [`Target`][genjax.inference.Target] change induced by a change to the arguments:
            ```python exec="yes" source="material-block" session="core"
            from genjax import gen
            from genjax import normal
            from genjax import EmptyProblem
            from genjax import Diff
            from genjax import ChoiceMapBuilder as C
            from genjax import UpdateProblemBuilder as U


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
                    EmptyProblem(),
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

        An `UpdateProblem` denotes a function $tr \\mapsto (T, T')$ from traces to a pair of targets (the previous [`Target`][genjax.inference.Target] $T$, and the final [`Target`][genjax.inference.Target] $T'$).

        Several common types of moves can be requested via the `GenericProblem` type:

        ```python exec="yes" source="material-block" session="core"
        from genjax import GenericProblem

        g = GenericProblem(
            Diff.unknown_change((1.0,)),  # "Argdiffs"
            EmptyProblem(),  # Subproblem
        )
        ```

        Creating problem instances is also possible using the `UpdateProblemBuilder`:
        ```python exec="yes" html="true" source="material-block" session="core"
        from genjax import UpdateProblemBuilder as U

        g = U.g(
            Diff.unknown_change((3.0,)),  # "Argdiffs"
            EmptyProblem(),  # Subproblem
        )
        print(g.render_html())
        ```

        `GenericProblem` contains information about changes to the arguments of the generative function ([`Argdiffs`][genjax.core.Argdiffs]) and a subproblem which specifies an additional move to be performed. The subproblem can be a bonafide [`UpdateProblem`][genjax.core.UpdateProblem] itself, or a [`Constraint`][genjax.core.Constraint] (like [`ChoiceMap`][genjax.core.ChoiceMap]).

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
        sample: "genjax.ChoiceMap",
        args: Arguments,
    ) -> tuple[Score, R]:
        """
        Return [the score][genjax.core.Trace.get_score] and [the return value][genjax.core.Trace.get_retval] when the generative function is invoked with the provided arguments, and constrained to take the provided sample as the sampled value.

        It is an error if the provided sample value is off the support of the distribution over the `Sample` type, or otherwise induces a partial constraint on the execution of the generative function (which would require the generative function to provide an `update` implementation which responds to the `UpdateProblem` induced by the [`importance`][genjax.core.GenerativeFunction.importance] interface).

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

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
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

        Under the hood, creates an [`UpdateProblem`][genjax.core.UpdateProblem] which requests that the generative function respond with a move from the _empty_ trace (the only possible value for _empty_ target $\\delta_\\emptyset$) to the target induced by the generative function for constraint $C$ with arguments $a$.
        """
        tr, w, _, _ = self.update(
            key,
            EmptyTrace(self),
            GenericProblem(Diff.unknown_change(args), ImportanceProblem(constraint)),
        )
        return tr, w

    @typecheck
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
        reverse: bool = False,
        unroll: int | bool = 1,
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

            reverse: optional boolean specifying whether to run the scan iteration forward (the default) or in reverse, equivalent to reversing the leading axes of the arrays in both `xs` and in `ys`.

            unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many scan iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

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

        return genjax.scan(n=n, reverse=reverse, unroll=unroll)(self)

    def accumulate(
        self, /, *, reverse: bool = False, unroll: int | bool = 1
    ) -> "GenerativeFunction[R]":
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

        Args:
            reverse: optional boolean specifying whether to run the accumulation forward (the default) or in reverse, equivalent to reversing the leading axes of the arrays in both `xs` and in `carries`.

            unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

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

        return genjax.accumulate(reverse=reverse, unroll=unroll)(self)

    def reduce(
        self, /, *, reverse: bool = False, unroll: int | bool = 1
    ) -> "GenerativeFunction[R]":
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

        Args:
            reverse: optional boolean specifying whether to run the accumulation forward (the default) or in reverse, equivalent to reversing the leading axis of the array `xs`.

            unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

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

        return genjax.reduce(reverse=reverse, unroll=unroll)(self)

    def iterate(self, /, *, n: Int, unroll: int | bool = 1) -> "GenerativeFunction[R]":
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

            unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

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

        return genjax.iterate(n=n, unroll=unroll)(self)

    def iterate_final(
        self, /, *, n: Int, unroll: int | bool = 1
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

            unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

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

        return genjax.iterate_final(n=n, unroll=unroll)(self)

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

    def or_else(
        self, gen_fn: "GenerativeFunction[S]", /
    ) -> "GenerativeFunction[R | S]":
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

    # TODO for this to be correct we need to either figure out how to splat
    # a bunch of type params, or change switch to have one return value.
    def switch(self, *branches: "GenerativeFunction[R]") -> "GenerativeFunction[R]":
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

    # TODO mix should also force the same return type.
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
    ) -> "GenerativeFunction":
        from genjax import Selection, marginal

        if selection is None:
            selection = Selection.all()

        return marginal(selection=selection, algorithm=algorithm)(self)


# NOTE: Setup a global handler stack for the `trace` callee sugar.
# C.f. above.
# This stack will not interact with JAX tracers at all
# so it's safe, and will be resolved at JAX tracing time.
GLOBAL_TRACE_OP_HANDLER_STACK: list[Callable[..., Any]] = []


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
    wrapped: GenerativeFunction

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
    def update(self, key: PRNGKey, trace: Trace, update_problem: GenericProblem):
        (argdiffs, _kwargdiffs) = update_problem.argdiffs
        return self.wrapped.update(
            key, trace, GenericProblem(argdiffs, update_problem.subproblem)
        )


@Pytree.dataclass
class GenerativeFunctionClosure(Generic[R], GenerativeFunction[R]):
    gen_fn: GenerativeFunction[R]
    args: tuple
    kwargs: dict

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
        args: tuple,
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
        update_problem: UpdateProblem,
    ) -> tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case GenericProblem(argdiffs, subproblem):
                full_argdiffs = (*self.args, *argdiffs)
                if self.kwargs:
                    maybe_kwarged_gen_fn = self.get_gen_fn_with_kwargs()
                    return maybe_kwarged_gen_fn.update(
                        key,
                        trace,
                        GenericProblem(
                            (full_argdiffs, self.kwargs),
                            subproblem,
                        ),
                    )
                else:
                    return self.gen_fn.update(key, trace, update_problem)
            case _:
                raise NotImplementedError

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: "genjax.ChoiceMap",
        args: tuple,
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
