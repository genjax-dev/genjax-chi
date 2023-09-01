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
from jax._src import core
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax.interpreters import ad as jax_autodiff
from jax.interpreters.ad import JVPTrace
from jax.interpreters.ad import JVPTracer

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import context as ctx
from genjax._src.core.interpreters import primitives
from genjax._src.core.interpreters import staging
from genjax._src.core.interpreters.context import Cont
from genjax._src.core.interpreters.context import Context
from genjax._src.core.pytree.pytree import Pytree
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
from genjax._src.core.typing import dispatch
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
    def enum_exact(self, key, primals, tangents, kont):
        pass


@dataclasses.dataclass
class SupportsMVD(Pytree):
    @abc.abstractmethod
    def mvd_estimate(self, key, primals, tangents, kont):
        pass


@dataclasses.dataclass
class SupportsCustom(Pytree):
    @abc.abstractmethod
    def custom_jvp_estimate(self, key, primals, tangents, kont):
        pass


#############
# ADEV term #
#############


@dataclasses.dataclass
class ADEVTerm(Pytree):
    @functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
    @abc.abstractmethod
    def sample(self, key, args):
        pass

    @abc.abstractmethod
    def jvp_estimate(self, key, primals, tangents, kont=identity):
        pass

    def jvp_estimate_nokwargs(self, key, primals, tangents):
        return self.jvp_estimate(key, primals, tangents)

    def grad_estimate(self, key, args):
        @typecheck
        def _inner(key: PRNGKey, args: Tuple):
            primal_tree = jtu.tree_structure(args)
            # Second args slot is for tangents.
            _, in_tree = jtu.tree_flatten((key, args, args))
            fun = lu.wrap_init(self.jvp_estimate_nokwargs)
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
            # Flat known is the key and the arguments.
            flat_known = jtu.tree_map(
                pe.PartialVal.known,
                jtu.tree_leaves((key, args)),
            )

            # Flat unknown is the tangents.
            flat_unknown = jtu.tree_map(
                lambda v: pe.PartialVal.unknown(core.ShapedArray(v.shape, v.dtype)),
                jtu.tree_leaves(args),
            )
            in_pvals = [*flat_known, *flat_unknown]
            jaxpr, out_pvals, consts = pe.trace_to_jaxpr_nounits(flat_fun, in_pvals)
            primal_dummies = [
                jax_autodiff.UndefinedPrimal(v.aval) for v in jaxpr.invars
            ]
            flat_args_bar = jax_autodiff.backward_pass(
                jaxpr, (), None, consts, primal_dummies, (1.0,)
            )
            args_bar = jtu.tree_unflatten(primal_tree, flat_args_bar)
            return args_bar

        return _inner(key, args)

    @sample.defjvp
    def sample_jvp(self, key, primals, tangents):
        primals, tangents = self.jvp_estimate(key, primals, tangents)
        return primals, tangents

    def __call__(self, key, *args):
        return self.sample(key, args)


###################
# ADEV primitives #
###################


@dataclasses.dataclass
class ADEVPrimitive(ADEVTerm):
    def flatten(self):
        return (), ()

    @abc.abstractmethod
    def sample(self, key, args):
        pass

    @abc.abstractmethod
    def jvp_estimate(self, key, primals, tangents, kont=identity):
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
        return prim.enum_exact(key, primals, tangents, kont)


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
    return adev_tag(strategy, meta=addr)


####################
# Sample intrinsic #
####################

sample_p = primitives.InitialStylePrimitive("sample")


def _abstract_adev_term_call(adev_term, strategy, *args):
    # Only the type matters here.
    key = jax.random.PRNGKey(0)
    v = adev_term.sample(key, args)
    return v


def _sample(adev_term, strategy, args):
    return primitives.initial_style_bind(sample_p)(_abstract_adev_term_call)(
        adev_term,
        strategy,
        *args,
    )


@typecheck
def sample(adev_term: ADEVTerm, args: Tuple, strategy=GradStratREINFORCE()):
    if isinstance(adev_term, ADEVPrimitive):
        # Embed using sow.
        strategy = strat(strategy, "sample")
    return _sample(adev_term, strategy, args)


##############
# Transforms #
##############


#####
# Sample
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
class SampleContext(ADEVContext):
    key: PRNGKey

    def flatten(self):
        return (self.key,), ()

    @classmethod
    def new(cls, key: PRNGKey):
        return SampleContext(key)

    def yield_state(self):
        return ()

    def handle_sample(self, _, *tracers, **params):
        in_tree = params.get("in_tree")
        adev_term, _, *args = jtu.tree_unflatten(in_tree, tracers)
        args = tuple(args)
        self.key, sub_key = jax.random.split(self.key)
        v = adev_term.sample(sub_key, args)
        return jtu.tree_leaves(v)


def sample_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, *args):
        context = SampleContext.new(key)
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
            primals, tangents = custom_rule(self, *tracers, **params)
            outvals = [JVPTracer(self, x, t) for x, t in zip(primals, tangents)]
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
    primals: Iterable[Any],
    tangents: Iterable[Any],
):
    """A context transformation that returns stateful context values."""
    trace = ADEVTrace(main, jc.cur_sublevel())
    in_tracers = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
    with staging.new_dynamic_context(main, context):
        context.trace = trace
        out_tracers = yield in_tracers, {}
        out_tracers = jax_util.safe_map(trace.full_raise, out_tracers)
        del main
    out_values = jtu.tree_map(lambda x: x.primal, out_tracers)
    out_tangents = jtu.tree_map(lambda x: x.tangent, out_tracers)
    yield (out_values, out_tangents), ()


# Designed to support ADEV - here, we enforce that primals and tangents
# must have the same Pytree shape.
def cps_jvp(f, ctx: Context):
    # Runs the interpreter.
    def _run_interpreter(main, kont, *args, **kwargs):
        with Cont.new() as interpreter:
            return interpreter.run_interpreter(
                ADEVTrace, main, kont, f, *args, **kwargs
            )

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(primals, tangents, kont, **kwargs):
        with jc.new_main(ADEVTrace) as main:
            fun = lu.wrap_init(functools.partial(_run_interpreter, main, kont), kwargs)
            flat_primals, primal_tree = jtu.tree_flatten(primals)
            flat_tangents, tangent_tree = jtu.tree_flatten(tangents)
            assert primal_tree == tangent_tree
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, primal_tree)
            flat_fun = _cps_jvp(flat_fun, main, ctx)
            (out_primals, out_tangents), _ = flat_fun.call_wrapped(
                flat_primals, flat_tangents
            )
            del main
            out_primals = jtu.tree_unflatten(out_tree(), out_primals)
            out_tangents = jtu.tree_unflatten(out_tree(), out_tangents)
        return (out_primals, out_tangents), ()

    return wrapped


@dataclasses.dataclass
class GradEstimateContext(ADEVContext):
    key: PRNGKey

    def flatten(self):
        return (self.key,), ()

    @classmethod
    def new(cls, key: PRNGKey):
        return GradEstimateContext(key)

    def yield_state(self):
        return ()

    def handle_sample(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        kont = params["kont"]
        adev_term, strategy, *tracers = jtu.tree_unflatten(in_tree, tracers)

        # TODO: generalize to Pytrees.
        primals, tangents = jax_util.unzip2((t.primal, t.tangent) for t in tracers)
        self.key, sub_key = jax.random.split(self.key)

        # Check if the term is an `ADEVProgram`, then use
        # `jvp_estimate` to propagate dual numbers.
        if isinstance(adev_term, ADEVProgram):
            primals, tangents = adev_term.jvp_estimate(sub_key, primals, tangents, kont)
            return primals, tangents

        # We're dealing with an `ADEVPrimitive` - we defer propagating
        # dual numbers to the gradient strategy.
        else:

            # This continuation accepts raw primals and tangents (as lists of arrays)
            # and then zips them into JVPTracer types -- allowing us to integrate our AD
            # with JAX's native forward mode AD.
            def flat_kont(primals: List, tangents: List):
                new_tracers = [
                    JVPTracer(self.trace, x, t) for x, t in zip(primals, tangents)
                ]
                # Loss must be a scalar, only a single retdual.
                (retdual,) = kont(*new_tracers)
                return retdual.primal, retdual.tangent

            # Defers handling the sample to the gradient strategy.
            # The gradient strategy will attempt to invoke the corresponding
            # method on the term, e.g. `reinforce_estimate`.
            primals, tangents = strategy.apply(
                adev_term,
                sub_key,
                primals,
                tangents,
                flat_kont,
            )
            return primals, tangents


def jvp_estimate_transform(source_fn, kont, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, primals, tangents):
        ctx = GradEstimateContext.new(key)
        (out_primals, out_tangents), _ = cps_jvp(source_fn, ctx)(
            primals, tangents, kont, **kwargs
        )
        # TODO: why is `tuple` conversion necessary here?
        return out_primals, out_tangents

    return wrapper


#################
# ADEV programs #
#################


@dataclasses.dataclass
class ADEVProgram(ADEVTerm):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    def sample(self, key, args):
        return sample_transform(self.source)(key, *args)

    def jvp_estimate(self, key, primals, tangents, kont=identity):
        return jvp_estimate_transform(self.source, kont)(key, primals, tangents)


@dispatch
def adev_convert(gen_fn: GenerativeFunction):
    """Overload for custom generative functions to support a conversion
    transformation to an ADEVProgram. Typically not invoked directly by a user,
    but is instead invoked by the `lang` decorator.

    Should return a `Callable`, which gets wrapped in `ADEVProgram` by `lang`.
    """
    raise NotImplementedError


@typecheck
def lang(gen_fn: GenerativeFunction):
    """Convert a `GenerativeFunction` to an `ADEVProgram`."""
    prim = registry.get(type(gen_fn))
    if prim is None:
        return ADEVProgram(adev_convert(gen_fn))
    else:
        return prim()


###########################
# ADEV primitive registry #
###########################

registry: Dict[Type[GenerativeFunction], Type[ADEVPrimitive]] = {}


def register(tg: Type[GenerativeFunction], prim: Type[ADEVPrimitive]):
    registry[tg] = prim
