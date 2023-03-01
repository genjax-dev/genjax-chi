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
import functools

import jax.core as jc
import jax.tree_util as jtu
from jax import api_util
from jax import linear_util as lu
from jax import util as jax_util

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import primitives
from genjax._src.core.interpreters import staging
from genjax._src.core.interpreters.context import Cont
from genjax._src.core.interpreters.context import Context
from genjax._src.core.interpreters.context import ContextualTrace
from genjax._src.core.interpreters.context import ContextualTracer
from genjax._src.core.pytree import Pytree
from genjax._src.core.transforms import harvest
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Iterable
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck


#############################
# Probabilistic computation #
#############################


@dataclasses.dataclass
class AbstractProbabilisticComputation(Pytree):
    def simulate(self, key, args):
        pass

    def grad_estimate(self, key, args):
        pass


# Overloads (for now, just distributions).
@dataclasses.dataclass
class ProbabilisticPrimitive(AbstractProbabilisticComputation):
    distribution: GenerativeFunction

    def simulate(self, key, args):
        return self.distribution.random_weighted(key, *args)


@dataclasses.dataclass
class ProbabilisticComputation(AbstractProbabilisticComputation):
    source: Callable

    def flatten(self):
        return (), (self.source,)


@typecheck
def prob_comp(gen_fn: GenerativeFunction):
    # `GenerativeFunction.adev_convert` is a function which expresses
    # forward sampling in terms of ADEV language primitives.
    return ProbabilisticComputation(gen_fn.adev_convert)


#######################
# Gradient strategies #
#######################

# TODO: determine the interface for these strategies.
# E.g. strategy gets `kont`, gets to choose how to use it.


@dataclasses.dataclass
class GradientStrategy(Pytree):
    pass


@dataclasses.dataclass
class Reinforce(GradientStrategy):
    pass


@dataclasses.dataclass
class ExactEnum(GradientStrategy):
    pass


######################
# Strategy intrinsic #
######################

# NOTE: this lets us embed strategies into code, and plant/change them
# via a transformation.

# TODO: To support address/hierarchy in strategies - we'll have to use
# a nest primitive from `harvest`.

NAMESPACE = "adev_strategy"
adev_tag = functools.partial(harvest.sow, tag=NAMESPACE)


@typecheck
def strat(strategy: GradientStrategy, addr):
    return adev_tag(strategy, name=addr)


####################
# Sample intrinsic #
####################

sample_p = primitives.InitialStylePrimitive("sample")


def _abstract_prob_comp_call(prob_comp, *args):
    return prob_comp.simulate(*args)


def _sample(prob_comp, strat, key, args, **kwargs):
    return primitives.initial_style_bind(sample_p)(_abstract_prob_comp_call)(
        prob_comp, key, strat, args, **kwargs
    )


@typecheck
def sample(
    prob_comp: AbstractProbabilisticComputation, key: PRNGKey, args: Tuple, **kwargs
):
    # Default strategy is REINFORCE.
    strategy = strat(Reinforce(), "sample")
    return _sample(prob_comp, key, strategy, args, **kwargs)


##############
# Transforms #
##############

#####
# Simulate
#####


@dataclasses.dataclass
class ADEVContext(Context):
    @abc.abstractmethod
    def handle_sample(self, *tracers, **params):
        pass

    def can_process(self, primitive):
        return False

    def process_primitive(self, primitive):
        raise NotImplementedError

    def get_custom_rule(self, primitive):
        if primitive is sample_p:
            return self.handle_trace
        else:
            return None


@dataclasses.dataclass
class SimulateContext(ADEVContext):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return SimulateContext()

    def yield_state(self):
        return ()

    def handle_sample(self, _, *args, **params):
        in_tree = params.get("in_tree")
        prob_comp, key, _, args = jtu.tree_unflatten(in_tree, args)
        key, v = prob_comp.simulate(key, args)
        return key, jtu.tree_leaves(v)


def simulate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(*args):
        ctx = SimulateContext.new()
        retvals, _ = ctx.transform(source_fn, ctx)(*args, **kwargs)
        return retvals

    return wrapper


#####
# Grad estimate transform
#####

# Real tangents & CPS


@lu.transformation
def _cps_jvp(
    main: jc.MainTrace, ctx: Context, primals: Iterable[Any], tangents: Iterable[Any]
):
    """A context transformation that returns stateful context values."""
    trace = ContextualTrace(main, jc.cur_sublevel())
    in_tracers = [ContextualTracer(trace, x, t) for x, t in zip(primals, tangents)]
    with staging.new_dynamic_context(main, ctx):
        ans = yield in_tracers, {}
        out_tracers = jax_util.safe_map(trace.full_raise, ans)
        stateful_tracers = jtu.tree_map(trace.full_raise, ctx.yield_state())
        del main
    (
        out_values,
        stateful_values,
    ) = jtu.tree_map(lambda x: x.val, (out_tracers, stateful_tracers))
    out_tangents = jtu.tree_map(lambda x: x.meta, out_tracers)
    yield (out_values, out_tangents), stateful_values


# Designed to support ADEV - here, we enforce that primals and tangents
# must have the same Pytree shape.
def cps_jvp(f, ctx: Context):
    # Runs the interpreter.
    def _run_interpreter(main, kont, *args, **kwargs):
        with Cont.new() as interpreter:
            return interpreter(main, kont, f, *args, **kwargs)

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(kont, primals, tangents, **kwargs):
        with jc.new_main(ContextualTrace) as main:
            fun = lu.wrap_init(functools.partial(_run_interpreter, main, kont), kwargs)
            flat_primals, primal_tree = jtu.tree_flatten(primals)
            flat_tangents, tangent_tree = jtu.tree_flatten(tangents)
            assert primal_tree == tangent_tree
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, primal_tree)
            flat_fun = _cps_jvp(flat_fun, main, ctx)
            (out_primals, out_tangents), ctx_statefuls = flat_fun.call_wrapped(
                flat_primals, flat_tangents
            )
            del main
        return (
            jtu.tree_unflatten(out_tree(), out_primals),
            jtu.tree_unflatten(out_tree(), out_tangents),
        ), ctx_statefuls

    return wrapped


@dataclasses.dataclass
class GradEstimateContext(ADEVContext):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return GradEstimateContext()

    def yield_state(self):
        return ()

    def handle_sample(self, _, *args, **params):
        in_tree = params["in_tree"]
        cont = params["cont"]
        prob_comp, key, strat, args = jtu.tree_unflatten(in_tree, args)
        key, v = strat.apply(cont, prob_comp, key, args)
        return key, jtu.tree_leaves(v)


def grad_estimate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, primals, tangents):
        ctx = GradEstimateContext.new()
        retvals, _ = cps_jvp(source_fn, ctx)(key, primals, tangents, **kwargs)
        return retvals

    return wrapper


##############
# Shorthands #
##############


def prob_comp(gen_fn: GenerativeFunction) -> ProbabilisticComputation:
    """Create an `adev.ProbabilisticComputation` from a generative function by
    wrapping the generative function's `adev_convert`."""
    return ProbabilisticComputation(gen_fn.adev_convert)
