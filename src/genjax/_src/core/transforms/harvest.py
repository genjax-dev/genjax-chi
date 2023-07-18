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

import dataclasses
import functools

import jax.core as jc
import jax.tree_util as jtu
import jax.util as jax_util
from jax import api_util
from jax import linear_util as lu
from jax._src import effects
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir

from genjax._src.core.interpreters import context
from genjax._src.core.interpreters import primitives as prim
from genjax._src.core.interpreters import staging
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FrozenSet
from genjax._src.core.typing import Hashable
from genjax._src.core.typing import Iterable
from genjax._src.core.typing import List
from genjax._src.core.typing import Optional
from genjax._src.core.typing import String
from genjax._src.core.typing import Union
from genjax._src.core.typing import Value


#################
# Sow intrinsic #
#################


sow_p = jc.Primitive("sow")
sow_p.multiple_results = True


class SowEffect(effects.Effect):
    __repr__ = lambda _: "Sow"


sow_effect = SowEffect()

effects.remat_allowed_effects.add_type(SowEffect)
effects.control_flow_allowed_effects.add_type(SowEffect)
effects.lowerable_effects.add_type(SowEffect)


@sow_p.def_impl
def _sow_impl(*args, **_):
    return args


@sow_p.def_effectful_abstract_eval
def _sow_abstract_eval(*avals, **_):
    return avals, {sow_effect}


def _sow_jvp(primals, tangents, **kwargs):
    out_primals = sow_p.bind(*primals, **kwargs)
    return out_primals, tangents


ad.primitive_jvps[sow_p] = _sow_jvp


def _sow_transpose(cts_in, *args, **kwargs):
    del args, kwargs
    return cts_in


ad.primitive_transposes[sow_p] = _sow_transpose


def _sow_batch_rule(batched_args, batch_dims, **params):
    outs = sow_p.bind(*batched_args, **params)
    return outs, batch_dims


batching.primitive_batchers[sow_p] = _sow_batch_rule
mlir.register_lowering(sow_p, lambda c, *args, **kw: args)


def sow(value, *, tag: Hashable, name: String, mode: String = "strict", key=None):
    """Marks a value with a name and a tag.

    Args:
      value: A JAX value to be tagged and named.
      tag: a string representing the tag of the sown value.
      name: a string representing the name to sow the value with.
      mode: The mode by which to sow the value. There are three options: 1.
        `'strict'` - if another value is sown with the same name and tag in the
        same context, harvest will throw an error. 2. `'clobber'` - if another is
        value is sown with the same name and tag, it will replace this value 3.
        `'append'` - sown values of the same name and tag are appended to a
        growing list. Append mode assumes some ordering on the values being sown
        defined by data-dependence.
      key: an optional JAX value that will be tied into the sown value.

    Returns:
      The original `value` that was passed in.
    """
    value = jtu.tree_map(jc.raise_as_much_as_possible, value)
    if key is not None:
        value = prim.tie_in(key, value)
    flat_args, in_tree = jtu.tree_flatten(value)
    out_flat = sow_p.bind(*flat_args, name=name, tag=tag, mode=mode, tree=in_tree)
    return jtu.tree_unflatten(in_tree, out_flat)


##################
# Nest intrinsic #
##################

nest_p = jc.CallPrimitive("nest")


def _nest_impl(f, *args, **_):
    with jc.new_sublevel():
        return f.call_wrapped(*args)


nest_p.def_impl(_nest_impl)


def _nest_lowering(ctx, *args, name, call_jaxpr, scope, **_):
    return mlir._xla_call_lower(  # pylint: disable=protected-access
        ctx,
        *args,
        name=jax_util.wrap_name(name, f"nest[{scope}]"),
        call_jaxpr=call_jaxpr,
        donated_invars=(False,) * len(args),
    )


mlir.register_lowering(nest_p, _nest_lowering)


def _nest_transpose_rule(*args, **kwargs):
    return ad.call_transpose(nest_p, *args, **kwargs)


ad.primitive_transposes[nest_p] = _nest_transpose_rule


def nest(f, *, scope: str):
    """Wraps a function to create a new scope for harvested values.

    Harvested values live in one dynamic name scope (for a particular tag),
    and in strict mode, values with the same name cannot be collected or injected
    more than once. `nest(f, scope=[name])` will take all tagged values in `f` and
    put them into a nested dictionary with key `[name]`. This enables having
    duplicate names in one namespace provided they are in different scopes. This
    is different from using a separate tag to namespace, as it enables creating
    nested/hierarchical structure within a single tag's namespace.
    Example:
    ```python
    def foo(x):
      return sow(x, tag='test', name='x')
    harvest(foo, tag='test')({}, 1.)  # (1., {'x': 1.})
    harvest(nest(foo, scope='a'), tag='test')({}, 1.)  # (1., {'a': {'x': 1.}})
    ```
    Args:
      f: a function to be transformed
      scope: a string that will act as the parent scope of all values tagged in
        `f`.
    Returns:
      A semantically identical function to `f`, but when harvested, uses nested
      values according to the input scope.
    """

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        out_flat = nest_p.bind(
            flat_fun, *flat_args, scope=scope, name=getattr(f, "__name__", "<no name>")
        )
        return jtu.tree_unflatten(out_tree(), out_flat)

    return wrapped


##########################
# Harvest transformation #
##########################


class HarvestTracer(context.ContextualTracer):
    """A `HarvestTracer` just encapsulates a single value."""

    def __init__(self, trace: "HarvestTrace", val: Value):
        self._trace = trace
        self.val = val

    @property
    def aval(self):
        return jc.raise_to_shaped(jc.get_aval(self.val))

    def full_lower(self):
        return self


class HarvestTrace(jc.Trace):
    """An evaluating trace that dispatches to a dynamic context."""

    def pure(self, val: Value) -> HarvestTracer:
        return HarvestTracer(self, val)

    def sublift(self, tracer: HarvestTracer) -> HarvestTracer:
        return self.pure(tracer.val)

    def lift(self, val: Value) -> HarvestTracer:
        return self.pure(val)

    def process_primitive(
        self,
        primitive: jc.Primitive,
        tracers: List[HarvestTracer],
        params: Dict[str, Any],
    ) -> Union[HarvestTracer, List[HarvestTracer]]:
        context = staging.get_dynamic_context(self)
        custom_rule = context.get_custom_rule(primitive)
        if custom_rule:
            return custom_rule(self, *tracers, **params)
        return self.default_process_primitive(primitive, tracers, params)

    def default_process_primitive(
        self,
        primitive: jc.Primitive,
        tracers: List[HarvestTracer],
        params: Dict[String, Any],
    ) -> Union[HarvestTracer, List[HarvestTracer]]:
        context = staging.get_dynamic_context(self)
        vals = [t.val for t in tracers]
        if primitive is sow_p:
            outvals = context.process_sow(*vals, **params)
            return jax_util.safe_map(self.pure, outvals)
        subfuns, params = primitive.get_bind_params(params)
        args = subfuns + vals
        outvals = primitive.bind(*args, **params)
        if not primitive.multiple_results:
            outvals = [outvals]
        out_tracers = jax_util.safe_map(self.pure, outvals)
        if primitive.multiple_results:
            return out_tracers
        return out_tracers[0]

    def process_call(
        self,
        call_primitive: jc.Primitive,
        f: Any,
        tracers: List[HarvestTracer],
        params: Dict[String, Any],
    ):
        context = staging.get_dynamic_context(self)
        if call_primitive is nest_p:
            return context.process_nest(self, f, *tracers, **params)
        return context.process_higher_order_primitive(
            self, call_primitive, f, tracers, params, False
        )

    def post_process_call(self, call_primitive, out_tracers, params):
        vals = tuple(t.val for t in out_tracers)
        master = self.main

        def todo(x):
            trace = HarvestTrace(master, jc.cur_sublevel())
            return jax_util.safe_map(functools.partial(HarvestTracer, trace), x)

        return vals, todo

    def process_map(
        self,
        call_primitive: jc.Primitive,
        f: Any,
        tracers: List[HarvestTracer],
        params: Dict[String, Any],
    ):
        context = staging.get_dynamic_context(self)
        return context.process_higher_order_primitive(
            self, call_primitive, f, tracers, params, True
        )

    post_process_map = post_process_call

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, *, symbolic_zeros):
        context = staging.get_dynamic_context(self)
        return context.process_custom_jvp_call(
            self, primitive, fun, jvp, tracers, symbolic_zeros=symbolic_zeros
        )

    def post_process_custom_jvp_call(self, out_tracers, jvp_was_run):
        context = staging.get_dynamic_context(self)
        return context.post_process_custom_jvp_call(self, out_tracers, jvp_was_run)

    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees):
        context = staging.get_dynamic_context(self)
        return context.process_custom_vjp_call(
            self, primitive, fun, fwd, bwd, tracers, out_trees
        )

    def post_process_custom_vjp_call(self, out_tracers, params):
        context = staging.get_dynamic_context(self)
        return context.post_process_custom_vjp_call(self, out_tracers, params)

    def post_process_custom_vjp_call_fwd(self, out_tracers, out_trees):
        context = staging.get_dynamic_context(self)
        return context.post_process_custom_vjp_call_fwd(self, out_tracers, out_trees)


@dataclasses.dataclass(frozen=True)
class HarvestSettings:
    """Contains the settings for a HarvestTrace."""

    tag: Hashable
    blocklist: FrozenSet[String]
    allowlist: Union[FrozenSet[String], None]
    exclusive: bool


@dataclasses.dataclass
class HarvestContext(context.Context):
    def get_custom_rule(self, primitive):
        return None

    def can_process(self, primitive):
        return primitive in [sow_p]

    def process_primitive(self, primitive, *args, **kwargs):
        if primitive is sow_p:
            return self.process_sow(*args, **kwargs)
        else:
            raise NotImplementedError

    def process_sow(self, *values, name, tag, mode, tree):
        """Handles a `sow` primitive in a `HarvestTrace`."""
        if mode not in {"strict", "append", "clobber"}:
            raise ValueError(f"Invalid mode: {mode}")
        if tag != self.settings.tag:
            if self.settings.exclusive:
                return values
            return sow_p.bind(*values, name=name, tag=tag, tree=tree, mode=mode)
        if self.settings.allowlist is not None and name not in self.settings.allowlist:
            return values
        if name in self.settings.blocklist:
            return values
        return self.handle_sow(*values, name=name, tag=tag, tree=tree, mode=mode)

    def handle_sow(self, *values, name, tag, mode, tree):
        raise NotImplementedError

    def process_nest(self, trace, f, *tracers, scope, name):
        raise NotImplementedError


###########
# Reaping #
###########


@dataclasses.dataclass
class Reap(Pytree):
    metadata: Dict[String, Any]
    value: Any

    def flatten(self):
        return (self.value,), (self.metadata,)

    @classmethod
    def new(cls, value, metadata):
        return Reap(metadata, value)


def unreap(v):
    def _unwrap(v):
        if isinstance(v, Reap):
            return v.value
        else:
            return v

    def _check(v):
        return isinstance(v, Reap)

    return jtu.tree_map(_unwrap, v, is_leaf=_check)


@dataclasses.dataclass
class ReapContext(HarvestContext):
    settings: HarvestSettings
    reaps: Dict[String, Any]

    def flatten(self):
        return (self.settings, self.reaps), ()

    @classmethod
    def new(cls, settings):
        reaps = dict()
        return ReapContext(settings, reaps)

    def yield_state(self):
        return (self.reaps,)

    def handle_sow(self, *values, name, tag, tree, mode):
        """Stores a sow in the reaps dictionary."""
        del tag
        if name in self.reaps and mode == "clobber":
            values, _ = jtu.tree_flatten(unreap(self.reaps[name]))
        elif name in self.reaps:
            raise ValueError(f"Variable has already been reaped: {name}")
        else:
            avals = jtu.tree_unflatten(
                tree,
                [jc.raise_to_shaped(jc.get_aval(v)) for v in values],
            )
            self.reaps[name] = Reap.new(
                jtu.tree_unflatten(tree, values),
                dict(mode=mode, aval=avals),
            )

        return values

    def process_nest(self, trace, f, *tracers, scope, name, **params):
        out_tracers, reap_tracers, _ = self.reap_higher_order_primitive(
            trace, nest_p, f, tracers, dict(params, name=name, scope=scope), False
        )
        tag = self.settings.tag
        if reap_tracers:
            flat_reap_tracers, reap_tree = jtu.tree_flatten(reap_tracers)
            trace.process_primitive(
                sow_p,
                flat_reap_tracers,
                dict(name=scope, tag=tag, tree=reap_tree, mode="strict"),
            )
        return out_tracers


def reap(
    fn,
    *,
    tag: Hashable,
    allowlist: Optional[Iterable[String]] = None,
    blocklist: Iterable[String] = frozenset(),
    exclusive: bool = False,
):
    blocklist = frozenset(blocklist)
    if allowlist is not None:
        allowlist = frozenset(allowlist)
    settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = ReapContext.new(settings)
        retvals, (reaps,) = context.transform(fn, ctx, HarvestTrace)(*args, **kwargs)
        return retvals, reaps

    return wrapper


############
# Planting #
############

plant_custom_rules = {}


@dataclasses.dataclass
class PlantContext(HarvestContext):
    """Contains the settings and storage for the current trace in the stack."""

    settings: HarvestSettings
    plants: Dict[String, Any]

    def flatten(self):
        return (self.plants,), (self.settings,)

    def __post_init__(self):
        self._already_planted = set()

    def yield_state(self):
        return ()

    def handle_sow(self, *values, name, tag, tree, mode):
        """Returns the value stored in the plants dictionary."""
        if name in self._already_planted and mode != "clobber":
            raise ValueError(f"Variable has already been planted: {name}")
        if name in self.plants:
            self._already_planted.add(name)
            return jtu.tree_leaves(self.plants[name])
        return sow_p.bind(*values, name=name, tag=tag, mode=mode, tree=tree)


def plant(
    fn,
    *,
    tag: Hashable,
    allowlist: Optional[Iterable[String]] = None,
    blocklist: Iterable[String] = frozenset(),
    exclusive: bool = False,
):
    blocklist = frozenset(blocklist)
    if allowlist is not None:
        allowlist = frozenset(allowlist)
    settings = HarvestSettings(tag, blocklist, allowlist, exclusive)

    @functools.wraps(fn)
    def wrapper(plants, *args, **kwargs):
        ctx = PlantContext.new(settings, plants)
        retvals, _ = context.transform(fn, ctx, HarvestTrace)(*args, **kwargs)
        return retvals

    return wrapper


#############
# Interface #
#############


def harvest(
    fn,
    *,
    tag: Hashable,
    allowlist: Optional[Iterable[String]] = None,
    blocklist: Iterable[String] = frozenset(),
    exclusive: bool = False,
):
    @functools.wraps(fn)
    def wrapper(plants, *args, **kwargs):
        f = plant(
            fn, tag=tag, allowlist=allowlist, blocklist=blocklist, exclusive=exclusive
        )
        f = reap(
            f, tag=tag, allowlist=allowlist, blocklist=blocklist, exclusive=exclusive
        )
        return f(plants, *args, **kwargs)

    return wrapper
