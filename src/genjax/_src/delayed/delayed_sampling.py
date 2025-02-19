# This file holds an interpreter for a probabilistic language which implements the delayed sampling algorithm.

from abc import abstractmethod
from dataclasses import dataclass, field
from functools import wraps

import jax.core as jc
import jax.random as jrand
import jax.tree_util as jtu
from jax import util as jax_util

from genjax._src.core.interpreters.forward import (
    Environment,
    InitialStylePrimitive,
    initial_style_bind,
    stage,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any, Generic, PRNGKey, TypeVar
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.generative_functions.distributions.tensorflow_probability import normal

R = TypeVar("R", bound=Any)

############
# Language #
############

# We extend `Jaxpr` with a set of primitives to model
# the checkpoints of Murray et al, 2017.
assume_p = InitialStylePrimitive("assume")
observe_p = InitialStylePrimitive("observe")
placeholder_p = InitialStylePrimitive("placeholder")


# These are the functions that users actually use to use the primitives
# in their code.
def assume(d: Distribution[R], *args):
    def sample_abstract(*args):
        key = jrand.key(0)
        _, v = d.random_weighted(key, *args)
        return v

    # Distributions are provided _as static values_ here
    # that means they should not hold JAX values!
    # They need to be nullary class instances!
    return initial_style_bind(assume_p, distribution=d)(sample_abstract)(*args)


def observe(v, d: Distribution[R], *args):
    def observe_abstract(v, *args):
        return v

    # Distributions are provided _as static values_ here
    # that means they should not hold JAX values!
    # They need to be nullary class instances!
    return initial_style_bind(observe_p, distribution=d)(observe_abstract)(v, *args)


####################################
# Delayed sampling virtual machine #
####################################

VarOrLiteral = jc.Var | jc.Literal


@dataclass
class DelayedSamplingVirtualMachine:
    """
    `DelayedSamplingVirtualMachine` implements the operations described by (Murray et al, 2017) to support delayed sampling.

    It keeps track of the dependency relationships between
    variables in the first order program representation (a `Jaxpr`).

    It also keeps track of the state necessary to implement the algorithm.
    """

    key: PRNGKey
    env: Environment = field(default_factory=Environment)
    initialized: list[jc.Var] = field(default_factory=list)
    marginalized: list[jc.Var] = field(default_factory=list)
    realized: list[jc.Var] = field(default_factory=list)
    parents: dict[int, jc.Var] = field(default_factory=dict)
    pullbacks: dict[int, "SymbolicPullback"] = field(default_factory=dict)
    children: dict[int, jc.Var] = field(default_factory=dict)
    incoming: dict[int, list[jc.Var | jc.Literal]] = field(default_factory=dict)

    def read(self, var: VarOrLiteral) -> Any:
        return self.env.read(var)

    def get(self, var: VarOrLiteral) -> Any:
        return self.env.get(var)

    def write(self, var: VarOrLiteral, cell: Any) -> Any:
        return self.env.write(var, cell)

    def __getitem__(self, var: VarOrLiteral) -> Any:
        return self.env.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        return var in self.env

    def maybe_add_edges(
        self,
        var: jc.Var,
        parents: list[jc.Var | jc.Literal],
    ):
        for u in parents:
            if isinstance(self.read(u), Delayed | Marginalized):
                assert isinstance(u, jc.Var)
                self.parents[var.count] = u
                self.children[u.count] = var

    def remove_parents(
        self,
        var: jc.Var,
    ):
        self.parents.pop(var.count)

    def remove_children(
        self,
        var: jc.Var,
    ):
        self.children.pop(var.count)

    def has_parent(
        self,
        var: jc.Var,
    ) -> bool:
        return var.count in self.parents

    def get_parent(
        self,
        var: jc.Var,
    ) -> jc.Var:
        return self.parents[var.count]

    def has_child(
        self,
        var: jc.Var,
    ) -> bool:
        return var.count in self.children

    def get_child(
        self,
        var: jc.Var,
    ) -> jc.Var:
        return self.children[var.count]

    def add_incoming(
        self,
        var: jc.Var,
        incoming: list[jc.Var | jc.Literal],
    ):
        self.incoming[var.count] = incoming

    def get_incoming(
        self,
        var: jc.Var,
    ):
        return tuple(self.read(v) for v in self.incoming[var.count])

    def check_terminal(
        self,
        var: jc.Var,
    ) -> bool:
        if not var in self.marginalized:
            return False
        child = self.children.get(var.count, var)
        return var == child or child not in self.marginalized

    def send_pullback(
        self,
        var: jc.Var,
        symbolic_pullback,
    ):
        parent = self.get_parent(var)
        self.pullbacks[parent.count] = symbolic_pullback

    def get_pullback(
        self,
        var: jc.Var,
    ) -> "SymbolicPullback":
        assert var.count in self.pullbacks
        return self.pullbacks[var.count]

    ######################################################
    # These can be read directly from the paper (Fig. 1) #
    ######################################################

    # These are the "local operations" as described in
    # Murray et al, 2017.

    def initialize(
        self,
        var: jc.Var,
        d: Distribution[R],
        incoming: list[jc.Var | jc.Literal],
    ):
        self.maybe_add_edges(var, incoming)
        if self.has_parent(var):
            self.initialized.append(var)
            val = Delayed(d)
        else:
            self.marginalized.append(var)
            invals = tuple(self.read(v) for v in incoming)
            val = Marginalized(d, invals)
        self.add_incoming(var, incoming)
        self.write(var, val)

    def marginalize(
        self,
        var: jc.Var,
    ):
        assert var in self.initialized and self.has_parent(var), (
            (var, var.count),
            self.initialized,
            self.parents,
        )
        delayed_value = self.read(var)
        incoming = make_symbolic(self.get_incoming(var))
        marginalized_value, symbolic_pullback = delayed_value.marginalize(*incoming)
        self.send_pullback(var, symbolic_pullback)
        self.initialized.remove(var)
        self.marginalized.append(var)
        self.write(var, marginalized_value)

    def sample(
        self,
        var: jc.Var,
    ):
        assert self.check_terminal(var)
        val = self.read(var)
        self.key, sub_key = jrand.split(self.key)
        sample = val.sample(sub_key)
        self.write(var, sample)
        self.realize(var)

    def observe(
        self,
        var: jc.Var,
        sample: jc.Var | jc.Literal,
    ):
        assert self.check_terminal(var), (var, self.children)
        value = self.read(var)
        sample_value = self.read(sample)
        weight = value.observe(sample_value)
        self.write(var, sample_value)
        self.realize(var)
        return weight

    def realize(
        self,
        var: jc.Var,
    ):
        assert self.check_terminal(var)
        self.marginalized.remove(var)
        self.realized.append(var)
        if self.has_parent(var):
            p = self.get_parent(var)
            sample_value = self.read(var)
            marginal_parent = self.read(p)
            parent_pullback = self.get_pullback(p)
            conditioned_parent = marginal_parent.pullback(sample_value, parent_pullback)
            self.write(p, conditioned_parent)
            self.remove_children(p)
        if self.has_child(var):
            u = self.get_child(var)
            self.marginalize(u)
            self.remove_children(var)

    def graft(
        self,
        var: jc.Var,
    ):
        if var in self.marginalized:
            if self.has_child(var):
                child = self.get_child(var)
                if child in self.marginalized:
                    self.prune(child)
        else:
            p = self.get_parent(var)
            self.graft(p)
            self.marginalize(var)
        assert self.check_terminal(var), (var.count, self.children)

    def prune(
        self,
        var: jc.Var,
    ):
        assert var in self.marginalized
        if self.has_child(var):
            child = self.get_child(var)
            if child in self.marginalized:
                self.prune(child)
        return self.sample(var)

    def value(
        self,
        var: jc.Literal | jc.Var,
    ):
        if isinstance(var, jc.Var):
            self.graft(var)
            self.sample(var)
        return self.read(var)


@Pytree.dataclass
class Marginalized(Generic[R], Pytree):
    """
    `Marginalized` is the type of values from a marginalized random variable. It keeps track of a distribution `Marginalized.d` and arguments `Marginalized.args` _which should never be `Delayed`_.

    Values of type `Marginalized` can be register observations, and can update themselves using `Marginalized.pullback` with pullbacks (from nodes that depend on the value) provided by `DelayedSamplingVirtualMachine`.
    """

    d: Distribution[R]
    args: tuple[Any, ...]

    def sample(self, key):
        _, v = self.d.random_weighted(key, *self.args)
        return v

    def pullback(
        self,
        observation: R,
        child_pullback: "SymbolicPullback",
    ):
        symbolic_d = convert_fwd(self.d)
        new_symbolic_d, new_args = child_pullback(observation, (symbolic_d, self.args))
        d = convert_bwd(new_symbolic_d)
        return Marginalized(d, new_args)

    def observe(self, sample):
        key = jrand.key(1)  # stub
        return self.d.estimate_logpdf(key, sample, *self.args)


@Pytree.dataclass
class Delayed(Generic[R], Pytree):
    """
    `Delayed` is the type of values from a delayed random variable. It is a symbolic representation of set of values one would get if they sampled from `Delayed.d`.

    Values of type `Delayed` can only be marginalized by the `DelayedSamplingVirtualMachine` as part of its operation.
    Users should never see these values in their own code.
    """

    d: Distribution[R]

    def marginalize(
        self,
        *invals,
    ) -> "tuple[Marginalized[R], SymbolicPullback]":
        symbolic_d = convert_fwd(self.d)
        (new_symbolic_d, args), pullback = symbolic_d.marginalize(*invals)
        d = convert_bwd(new_symbolic_d)
        return Marginalized(d, args), pullback


@Pytree.dataclass
class DelayedSamplingInterpreter(Pytree):
    """
    `DelayedSamplingInterpreter` traverses the `Jaxpr` in a forward fashion, but defers most of the execution semantics of the delayed sampling algorithm to the `DelayedSamplingVirtualMachine`.

    The interpreter is responsible for reaching the "checkpoints" described in (Murray et al, 2017) -- and then defers local operations to the `DelayedSamplingVirtualMachine`.

    It also handles compatibility with "foreign" JAX functions (by forcing `DelayedSamplingVirtualMachine` to materialize the values required by those functions).
    """

    def _eval_jaxpr_delayed(
        self,
        key: PRNGKey,
        jaxpr: jc.Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = DelayedSamplingVirtualMachine(key)
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, args)
        for eqn in jaxpr.eqns:
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)

            # Assume checkpoint
            if eqn.primitive == assume_p:
                (outvar,) = eqn.outvars
                env.initialize(
                    outvar,
                    params["distribution"],
                    eqn.invars,
                )

            # Observe checkpoint
            elif eqn.primitive == observe_p:
                (outvar,) = eqn.outvars
                sample, *rst = eqn.invars
                env.initialize(outvar, params["distribution"], rst)
                env.graft(outvar)
                env.observe(outvar, sample)

            # Value checkpoints are implicit here...
            else:
                if eqn.primitive == placeholder_p:
                    # Do something special for deterministic primitives
                    # with known rules?
                    pass
                else:
                    # Force values.
                    invals = [env.value(var) for var in eqn.invars]
                    args = subfuns + invals
                    outvals = eqn.primitive.bind(*args, **params)
                    if not eqn.primitive.multiple_results:
                        outvals = [outvals]
                    jax_util.safe_map(env.write, eqn.outvars, outvals)

        # Value checkpoints are forced here.
        return [env.value(var) for var in jaxpr.outvars]

    def run_interpreter(self, key: PRNGKey, fn, *args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_delayed(
            key,
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


##########################################################
# Symbolic distributions with condition / marginal rules #
##########################################################

# This stuff is supposed to be exposed to users of this language,
# so they can add new primitive distributions + rules
# for marginalization and conditioning.

SymbolicBundle = tuple["SymbolicDistribution", Any]
SymbolicPullback = Any


class SymbolicDistribution(Pytree):
    """
    A `SymbolicDistribution` implements `marginalize`, which is a rule for marginalizing itself given a marginalized parent.
    """

    @abstractmethod
    def marginalize(
        self,
        *args,
    ) -> tuple[SymbolicBundle, SymbolicPullback]:
        raise NotImplementedError


@Pytree.dataclass
class SymbolicNormal(SymbolicDistribution):
    # TODO: can this be generalized to support multiple parents?
    # What part of delayed sampling breaks down when one wants
    # to do this?
    def marginalize(  # pyright: ignore
        self,
        mean,
        cov,
    ) -> tuple[SymbolicBundle, SymbolicPullback]:
        match (mean, cov):
            case ((SymbolicNormal(), _mean, _cov), _):
                # The pullback is a rule that allows us to
                # construct the posterior for the parent
                # given the marginal of the parent.
                def mean_pullback(
                    child_observation: Any,
                    parent_marginal: SymbolicBundle,
                ) -> SymbolicBundle:
                    _, args = parent_marginal
                    return (SymbolicNormal(), args)

                return (SymbolicNormal(), (_mean, cov + _cov)), mean_pullback
            case _, _:
                raise NotImplementedError


symbolic_convert: dict[Distribution[Any], SymbolicDistribution] = {
    normal: SymbolicNormal(),
}


def convert_fwd(d: Distribution[Any]) -> SymbolicDistribution:
    return symbolic_convert[d]


def convert_bwd(symd: SymbolicDistribution) -> Distribution[Any]:
    rev_symb = {value: key for key, value in symbolic_convert.items()}
    return rev_symb[symd]


def make_symbolic(vs: tuple[Any, ...]):
    def convert_marginalized_to_symbolic(v):
        return (convert_fwd(v.d), *v.args)

    return tuple(
        convert_marginalized_to_symbolic(v) if isinstance(v, Marginalized) else v
        for v in vs
    )


# Returns a function which runs the delayed sampling interpreter.
def delay(fn):
    @wraps(fn)
    def wrapped(key: PRNGKey, *args):
        interpreter = DelayedSamplingInterpreter()
        return interpreter.run_interpreter(
            key,
            fn,
            *args,
        )

    return wrapped
