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

import jax
import jax.core as jc
import jax.tree_util as jtu
from jax import api_util
from jax import linear_util as lu
from jax import util as jax_util
from jax.interpreters import ad as jax_autodiff
from jax.interpreters.ad import JVPTrace
from jax.interpreters.ad import JVPTracer

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import context as ctx
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


# Trivial continuation.
identity = lambda v: v

############################
# Gradient strategy traits #
############################


@dataclasses.dataclass
class SupportsREINFORCE(Pytree):
    @abc.abstractmethod
    def reinforce_estimate(self, key, primals, tangents, kont):
        pass


@dataclasses.dataclass
class SupportsEnum(Pytree):
    @abc.abstractmethod
    def enum_estimate(self, key, primals, tangents, kont):
        pass


@dataclasses.dataclass
class SupportsMVD(Pytree):
    @abc.abstractmethod
    def mvd_estimate(self, key, primals, tangents, kont):
        pass


@dataclasses.dataclass
class SupportsCustom(Pytree):
    @abc.abstractmethod
    def custom_grad_estimate(self, key, primals, tangents, kont):
        pass


#############
# ADEV term #
#############


@dataclasses.dataclass
class ADEVTerm(Pytree):
    @functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
    @abc.abstractmethod
    def simulate(self, key, args):
        pass

    @abc.abstractmethod
    def grad_estimate(self, key, primals, tangents, kont=identity):
        pass

    @simulate.defjvp
    def simulate_jvp(self, key, primals, tangents):
        key, primals, tangents = self.grad_estimate(key, primals, tangents)
        return (key, primals), tangents

    def __call__(self, key, *args):
        return self.simulate(key, args)


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
    def grad_estimate(self, key, primals, tangents, kont=identity):
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
class GradStratREINFORCE(GradientStrategy):
    def apply(self, prim, key, primals, tangents, kont):
        assert isinstance(prim, SupportsREINFORCE)
        return prim.reinforce_estimate(key, primals, tangents, kont)


@dataclasses.dataclass
class GradStratEnum(GradientStrategy):
    def apply(self, prim, key, primals, tangents, kont):
        assert isinstance(prim, SupportsEnum)
        return prim.enum_estimate(key, primals, tangents, kont)


@dataclasses.dataclass
class GradStratMVD(GradientStrategy):
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


def _abstract_adev_term_call(adev_term, key, strategy, *args):
    return adev_term.simulate(key, args)


def _sample(adev_term, key, strategy, args, **kwargs):
    return primitives.initial_style_bind(sample_p)(_abstract_adev_term_call)(
        adev_term, key, strategy, *args, **kwargs
    )


@typecheck
def sample(adev_term: ADEVTerm, key: PRNGKey, args: Tuple, **kwargs):
    # Default strategy is REINFORCE.
    strategy = GradStratREINFORCE()
    if isinstance(adev_term, ADEVPrimitive):
        # Embed using sow.
        strategy = strat(strategy, "sample")
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
            return self.handle_sample
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

    def handle_sample(self, _, *tracers, **params):
        in_tree = params.get("in_tree")
        adev_term, key, _, *args = jtu.tree_unflatten(in_tree, tracers)
        args = tuple(args)
        key, v = adev_term.simulate(key, args)
        return jtu.tree_leaves((key, v))


def simulate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(*args):
        context = SimulateContext.new()
        retvals, _ = ctx.transform(source_fn, context)(*args, **kwargs)
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
        if primitive is sample_p:
            custom_rule = context.get_custom_rule(primitive)
            assert custom_rule is not None
            # TODO: Pull key out of tracer.
            # Try and fix this in the interpreter somehow.
            key, *tracers = tracers
            key_primal = key.primal
            key, primals, tangents = custom_rule(self, key_primal, *tracers, **params)
            outvals = jtu.tree_leaves(
                (key, [JVPTracer(self, x, t) for x, t in zip(primals, tangents)])
            )
            return outvals
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
        kont = params.pop("kont")
        primal_out, tangent_out = jvp(primals_in, tangents_in, **params)
        if primitive.multiple_results:
            return kont(
                *[JVPTracer(self, x, t) for x, t in zip(primal_out, tangent_out)]
            )
        else:
            return kont(JVPTracer(self, primal_out, tangent_out))

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
    context: Context,
    key: PRNGKey,
    primals: Iterable[Any],
    tangents: Iterable[Any],
):
    """A context transformation that returns stateful context values."""
    trace = ADEVTrace(main, jc.cur_sublevel())
    in_tracers = [key] + [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
    with staging.new_dynamic_context(main, context):
        context.trace = trace
        key, *out_tracers = yield in_tracers, {}
        out_tracers = jax_util.safe_map(trace.full_raise, out_tracers)
        del main
    out_values = jtu.tree_map(lambda x: x.primal, out_tracers)
    out_tangents = jtu.tree_map(lambda x: x.tangent, out_tracers)
    yield (key, out_values, out_tangents), ()


# Designed to support ADEV - here, we enforce that primals and tangents
# must have the same Pytree shape.
def cps_jvp(f, ctx: Context):
    # Runs the interpreter.
    def _run_interpreter(main, kont, *args, **kwargs):
        with Cont.new() as interpreter:
            return interpreter(ADEVTrace, main, kont, f, *args, **kwargs)

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(key, primals, tangents, kont, **kwargs):
        with jc.new_main(ADEVTrace) as main:
            fun = lu.wrap_init(functools.partial(_run_interpreter, main, kont), kwargs)
            flat_primals, primal_tree = jtu.tree_flatten(primals)
            flat_tangents, tangent_tree = jtu.tree_flatten(tangents)
            assert primal_tree == tangent_tree
            _, args_tree = jtu.tree_flatten((key, primals))
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, args_tree)
            flat_fun = _cps_jvp(flat_fun, main, ctx)
            (key, out_primals, out_tangents), _ = flat_fun.call_wrapped(
                key, flat_primals, flat_tangents
            )
            del main
            _, out_primals = jtu.tree_unflatten(out_tree(), (key, out_primals))
            _, out_tangents = jtu.tree_unflatten(out_tree(), (key, out_tangents))
        return (key, out_primals, out_tangents), ()

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
        adev_term, key, strategy, *tracers = jtu.tree_unflatten(in_tree, tracers)

        # TODO: generalize to Pytrees.
        primals, tangents = jax_util.unzip2((t.primal, t.tangent) for t in tracers)

        # Check if the term is an `ADEVProgram`, then use
        # `grad_estimate` to propagate dual numbers.
        if isinstance(adev_term, ADEVProgram):
            key, primals, tangents = adev_term.grad_estimate(
                key, primals, tangents, kont
            )
            return key, primals, tangents

        # We're dealing with an `ADEVPrimitive` - we defer propagating
        # dual numbers to the gradient strategy.
        else:

            def lift_kont(key, primals, tangents):
                new_tracers = [
                    JVPTracer(self.trace, x, t) for x, t in zip(primals, tangents)
                ]
                key, retdual = kont(key, *new_tracers)
                return key, retdual.primal, retdual.tangent

            key, primals, tangents = strategy.apply(
                adev_term, key, primals, tangents, lift_kont
            )
            return key, primals, tangents


def grad_estimate_transform(source_fn, kont, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, primals, tangents):
        ctx = GradEstimateContext.new()
        (key, out_primals, out_tangents), _ = cps_jvp(source_fn, ctx)(
            key, primals, tangents, kont, **kwargs
        )
        # TODO: why is `tuple` conversion necessary here?
        return key, tuple(out_primals), tuple(out_tangents)

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

    def grad_estimate(self, key, primals, tangents, kont=identity):
        return grad_estimate_transform(self.source, kont)(key, primals, tangents)


@typecheck
def lang(gen_fn: GenerativeFunction):
    """Convert a `GenerativeFunction` to an `ADEVProgram`."""
    prim = registry.get(type(gen_fn))
    if prim is None:
        # `GenerativeFunction.adev_simulate` is an interface which expresses
        # forward sampling from a generative function in terms of
        # ADEV language primitives.
        return ADEVProgram(gen_fn.adev_simulate)
    else:
        return prim()


###########################
# ADEV primitive registry #
###########################

registry: Dict[Type[GenerativeFunction], Type[ADEVPrimitive]] = {}


def register(tg: Type[GenerativeFunction], prim: Type[ADEVPrimitive]):
    registry[tg] = prim
