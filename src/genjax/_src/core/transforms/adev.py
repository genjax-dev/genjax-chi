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
from jax.interpreters import ad as jax_autodiff
from jax.interpreters.ad import JVPTrace
from jax.interpreters.ad import JVPTracer

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import primitives
from genjax._src.core.interpreters import staging
from genjax._src.core.interpreters.context import Cont
from genjax._src.core.interpreters.context import Context
from genjax._src.core.pytree import Pytree
from genjax._src.core.transforms import harvest
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Iterable
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import String
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Type
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck


identity = lambda v: v

############################
# Gradient strategy traits #
############################


@dataclasses.dataclass
class SupportsReinforce(Pytree):
    @abc.abstractmethod
    def reinforce_estimate(self, key, duals, kont):
        pass


@dataclasses.dataclass
class SupportsEnum(Pytree):
    @abc.abstractmethod
    def enum_estimate(self, key, duals, kont):
        pass


@dataclasses.dataclass
class SupportsMVD(Pytree):
    @abc.abstractmethod
    def mvd_estimate(self, key, duals, kont):
        pass


#############
# ADEV term #
#############


@dataclasses.dataclass
class ADEVTerm(Pytree):
    @abc.abstractmethod
    def simulate(self, key, args):
        pass

    @abc.abstractmethod
    def grad_estimate(self, key, primals, tangents):
        pass


###################
# ADEV primitives #
###################


@dataclasses.dataclass
class ADEVPrimitive(ADEVTerm):
    def flatten(self):
        return (), ()

    @abc.abstractmethod
    def simulate(self, key, args):
        pass

    @abc.abstractmethod
    def grad_estimate(self, key, primals, tangents):
        pass


#######################
# Gradient strategies #
#######################


# Indicator classes.
@dataclasses.dataclass
class GradientStrategy(Pytree):
    def flatten(self):
        return (), ()

    @abc.abstractmethod
    def apply(self, prim, key, primals, tangents, kont):
        pass


@dataclasses.dataclass
class Reinforce(GradientStrategy):
    def apply(self, prim, key, primals, tangents, kont):
        assert isinstance(prim, SupportsReinforce)
        return prim.reinforce_estimate(key, primals, tangents, kont)


@dataclasses.dataclass
class Enum(GradientStrategy):
    def apply(self, prim, key, primals, tangents, kont):
        assert isinstance(prim, SupportsEnum)
        return prim.enum_estimate(key, primals, tangents, kont)


@dataclasses.dataclass
class MVD(GradientStrategy):
    def apply(self, prim, key, primals, tangents, kont):
        assert isinstance(prim, SupportsMVD)
        return prim.mvd_estimate(key, primals, tangents, kont)


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


def _abstract_adev_term_call(adev_term, *args):
    return adev_term.simulate(*args)


def _sample(adev_term, strat, key, args, **kwargs):
    return primitives.initial_style_bind(sample_p)(_abstract_adev_term_call)(
        adev_term, key, strat, args, **kwargs
    )


@typecheck
def sample(adev_term: ADEVTerm, key: PRNGKey, args: Tuple, **kwargs):
    # Default strategy is REINFORCE.
    strategy = strat(Reinforce(), "sample")
    return _sample(adev_term, key, strategy, args, **kwargs)


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
        adev_term, key, _, args = jtu.tree_unflatten(in_tree, args)
        key, v = adev_term.simulate(key, args)
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

# CPS with real tangents.
class ADEVTrace(JVPTrace):
    """A forward-mode AD trace that dispatches to a dynamic context."""

    def process_primitive(
        self,
        primitive: jc.Primitive,
        tracers: List[JVPTracer],
        params: Dict[String, Any],
    ) -> Union[JVPTracer, List[JVPTracer]]:
        context = staging.get_dynamic_context(self)
        custom_rule = context.get_custom_rule(primitive)
        if custom_rule:
            return custom_rule(self, *tracers, **params)
        params.pop("kont")
        return self.default_process_primitive(primitive, tracers, params)

    def default_process_primitive(
        self,
        primitive: jc.Primitive,
        tracers: List[JVPTracer],
        params: Dict[String, Any],
    ) -> Union[JVPTracer, List[JVPTracer]]:
        primals_in, tangents_in = jax_util.unzip2(
            (t.primal, t.tangent) for t in tracers
        )
        jvp = jax_autodiff.primitive_jvps.get(primitive)
        if not jvp:
            msg = f"Differentiation rule for '{primitive}' not implemented"
            raise NotImplementedError(msg)
        primal_out, tangent_out = jvp(primals_in, tangents_in, **params)
        if primitive.multiple_results:
            return [JVPTracer(self, x, t) for x, t in zip(primal_out, tangent_out)]
        else:
            return JVPTracer(self, primal_out, tangent_out)

    def process_call(
        self,
        call_primitive: jc.Primitive,
        f: Any,
        tracers: List[JVPTracer],
        params: Dict[str, Any],
    ):
        context = staging.get_dynamic_context(self)
        return context.process_higher_order_primitive(
            self, call_primitive, f, tracers, params, False
        )

    def post_process_call(self, call_primitive, out_tracers, params):
        vals = tuple(t.val for t in out_tracers)
        master = self.main

        def todo(x):
            trace = ADEVTrace(master, jc.cur_sublevel())
            return jax_util.safe_map(functools.partial(JVPTracer, trace), x)

        return vals, todo

    def process_map(
        self,
        call_primitive: jc.Primitive,
        f: Any,
        tracers: List[JVPTracer],
        params: Dict[str, Any],
    ):
        context = staging.get_dynamic_context(self)
        return context.process_higher_order_primitive(
            self, call_primitive, f, tracers, params, True
        )


@lu.transformation
def _cps_jvp(
    main: jc.MainTrace,
    ctx: Context,
    key: PRNGKey,
    primals: Iterable[Any],
    tangents: Iterable[Any],
):
    """A context transformation that returns stateful context values."""
    trace = ADEVTrace(main, jc.cur_sublevel())
    in_tracers = [key] + [JVPTrace(trace, x, t) for x, t in zip(primals, tangents)]
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
    def wrapped(key, primals, tangents, kont, **kwargs):
        with jc.new_main(ADEVTrace) as main:
            fun = lu.wrap_init(functools.partial(_run_interpreter, main, kont), kwargs)
            flat_primals, primal_tree = jtu.tree_flatten(primals)
            flat_tangents, tangent_tree = jtu.tree_flatten(tangents)
            assert primal_tree == tangent_tree
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, primal_tree)
            flat_fun = _cps_jvp(flat_fun, main, ctx)
            (key, out_primals, out_tangents), _ = flat_fun.call_wrapped(
                key, flat_primals, flat_tangents
            )
            del main
        return (
            key,
            jtu.tree_unflatten(out_tree(), out_primals),
            jtu.tree_unflatten(out_tree(), out_tangents),
        )

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

    def handle_sample(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        kont = params["kont"]
        adev_term, key, strat, tracers = jtu.tree_unflatten(in_tree, tracers)
        key, v = strat.apply(adev_term, key, tracers, kont)
        return key, jtu.tree_leaves(v)


def grad_estimate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, primals, tangents):
        ctx = GradEstimateContext.new()
        (_, out_tangents), _ = cps_jvp(source_fn, ctx)(key, primals, tangents, **kwargs)
        return out_tangents

    return wrapper


#################
# ADEV programs #
#################


@dataclasses.dataclass
class ADEVProgram(ADEVTerm):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    def simulate(self, key, args):
        return simulate_transform(self.source)(key, args)

    def grad_estimate(self, key, primals, tangents):
        return grad_estimate_transform(self.source)(key, primals, tangents)


@typecheck
def adev(gen_fn: GenerativeFunction):
    """Convert a `GenerativeFunction` to an `ADEVProgram`."""
    prim = registry.get(type(gen_fn))
    if prim is None:
        # `GenerativeFunction.adev_simulate` is an interface which expresses
        # forward sampling from a generative function in terms of
        # ADEV language primitives.
        return ADEVProgram(gen_fn.adev_simulate)
    else:
        return prim


###########################
# ADEV primitive registry #
###########################

registry: Dict[Type[GenerativeFunction], Type[ADEVPrimitive]] = {}


def register(tg: Type[GenerativeFunction], prim: Type[ADEVPrimitive]):
    registry[tg] = prim
