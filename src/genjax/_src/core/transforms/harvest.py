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
"""This module contains a general-purpose set of tools for transforming
functions with a specific side-effect mechanism into pure functions. The names
of the transformations in this module are inspired by the Sow/Reap mechanism in
Wolfram Mathematica.

The harvest module exposes two main functions: `sow` and `harvest`.

* `sow` is
used to tag values.
* `harvest` can inject values into functions or pull out
tagged values.

`harvest` is a very general purpose transformation purely focused on converting
functions that have special side-effects (defined using `sow`) and
"functionalizing" them.

Specifically, a function
`f :: (x: X) -> Y` has a set of defined intermediates, or `Sows`. This set
can be divided into intermediates you are "collecting" and intermediates you are
"injecting", or `Reaps` and `Plants` respectively. Functionalizing
`f` now gives you `harvest(f) :: (plants: Plants, x: X) -> Tuple[Y, Reaps]`.

Generally, most users will not need to use `harvest` directly, but will use
wrappers around it.
"""

import abc
import collections
import dataclasses
import functools

import jax.core as jc
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.util as jax_util
from jax import api_util
from jax import lax
from jax import linear_util as lu
from jax._src import ad_checkpoint
from jax._src import core as jax_core
from jax._src import effects
from jax._src import pjit
from jax._src import sharding_impls
from jax._src.lax import control_flow as lcf
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

from genjax._src.core.interpreters import context
from genjax._src.core.interpreters import staging
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Hashable
from genjax._src.core.typing import Iterable
from genjax._src.core.typing import List
from genjax._src.core.typing import String
from genjax._src.core.typing import Tuple
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


def sow(
    value: Any,
    *,
    tag: Hashable,
    meta: Any,
    mode: String = "strict",
) -> Any:
    """> Marks a value with a metadata value and a tag.

    `sow` is the function used to tag values in a function. It takes in a single
    positional argument, `value`, which is returned as an output, so `sow` outside
    of a tracing context behaves like the identity function, i.e.
    `sow(x, ...) == x`. It also takes in two mandatory keyword arguments,
    `tag` and `name`.

    * `tag` is a string used to namespace intermediate values in a
    function. For example, some intermediates may be useful
    for logging or debugging. The tag enables `harvest` to interact with only one set of intermediates at a time.

    * The `name` is a string that describes the value you are `sow`-ing. Eventually,
    when calling `harvest` on a function, the `name` is used as the identifier
    for the intermediate value.

    Finally, `sow` takes in an optional string keyword argument `mode`, which is by
    default set to `'strict'`. The `mode` of a `sow` describes how it behaves when
    the same name appears multiple times. In "strict" mode, `sow` will error if the
    same `(tag, name)` appears more than once. Another option is `'append'`, in
    which all sows of the same name will be appended into a growing array. Finally,
    there is `'clobber'`, where only the final sown value for a given `(tag, name)`
    will be returned. The final optional argument for `sow` is `key`, which will
    automatically be tied-in to the output of `sow` to introduce a fake
    data-dependence. By default, it is `None`.

    Args:
      value: A JAX value to be tagged and metad.
      tag: A `String` representing the tag of the sown value.
      meta: A piece of metadata to sow the value with.
      mode: The mode by which to sow the value. There are three options:

        * `'strict'` - if another value is sown with the same metadata and tag in the
        same context, harvest will throw an error.

        * `'clobber'` - if another is
        value is sown with the same meta and tag, it will replace this value

        * `'append'` - sown values of the same meta and tag are appended to a
        growing list. Append mode assumes some ordering on the values being sown
        defined by data-dependence.

    Returns:
        value: The original `value` that was passed in.
    """
    value = jtu.tree_map(jc.raise_as_much_as_possible, value)
    flat_args, in_tree = jtu.tree_flatten(value)
    out_flat = sow_p.bind(*flat_args, meta=meta, tag=tag, mode=mode, tree=in_tree)
    return jtu.tree_unflatten(in_tree, out_flat)


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

    def process_sow(self, *values, meta, tag, mode, tree):
        """Handles a `sow` primitive in a `HarvestTrace`."""
        if mode not in {"strict", "append", "clobber"}:
            raise ValueError(f"Invalid mode: {mode}")
        if tag != self.settings.tag:
            return sow_p.bind(*values, meta=meta, tag=tag, tree=tree, mode=mode)
        return self.handle_sow(*values, meta=meta, tag=tag, tree=tree, mode=mode)

    def handle_sow(self, *values, meta, tag, mode, tree):
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


def tree_unreap(v):
    def _unwrap(v):
        if isinstance(v, Reap):
            return v.value
        else:
            return v

    def _check(v):
        return isinstance(v, Reap)

    return jtu.tree_map(_unwrap, v, is_leaf=_check)


@dataclasses.dataclass
class ReapState(Pytree):
    @abc.abstractmethod
    def sow(self, values, tree, meta, mode):
        pass


reap_custom_rules = {}


@dataclasses.dataclass
class ReapContext(HarvestContext):
    settings: HarvestSettings
    reaps: ReapState

    def flatten(self):
        return (self.settings, self.reaps), ()

    @classmethod
    def new(cls, settings, reap_state):
        return ReapContext(settings, reap_state)

    def get_custom_rule(self, primitive):
        return reap_custom_rules.get(primitive)

    def yield_state(self):
        return (self.reaps,)

    def handle_sow(self, *values, meta, tag, tree, mode):
        """Stores a sow in the reaps dictionary."""
        values = self.reaps.sow(values, tree, meta, mode)
        del tag
        return values


def call_and_reap(
    f,
    *,
    tag: Hashable,
):
    """Transforms a function into one that additionally returns its sown
    values.

    Args:
      f: a function to be transformed.
      tag: a string tag; only sown values with `tag` will be reaped.

    Returns:
      A new function that executes the original and returns its sown values as
      an additional return value.
    """
    settings = HarvestSettings(tag)

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        with jc.new_main(HarvestTrace) as main:
            flat_fun = reap_function(flat_fun, main, settings, False)
            out_flat, reaps = flat_fun.call_wrapped(flat_args)
            del main
        return jtu.tree_unflatten(out_tree(), out_flat), reaps

    return wrapped


@lu.transformation
def reap_function(
    main: jc.MainTrace,
    settings: HarvestSettings,
    return_metadata: bool,
    args: Iterable[Any],
):
    """A function transformation that returns reap values."""
    trace = HarvestTrace(main, jc.cur_sublevel())
    in_tracers = jax_util.safe_map(trace.pure, args)
    context = ReapContext(settings, {})
    with staging.new_dynamic_context(main, context):
        ans = yield in_tracers, {}
        out_tracers = jax_util.safe_map(trace.full_raise, ans)
        reap_tracers = jtu.tree_map(
            lambda x: jtu.tree_map(trace.full_raise, x.value), context.reaps
        )
        reap_metadata = jtu.tree_map(lambda x: x.metadata, context.reaps)
        del main
    out_values, reap_values = jtu.tree_map(lambda x: x.val, (out_tracers, reap_tracers))
    if return_metadata:
        out = (out_values, reap_values, reap_metadata)
    else:
        out = (out_values, reap_values)
    yield out


def reap_eval(
    f: lu.WrappedFun,
    trace: HarvestTrace,
    settings: HarvestSettings,
) -> Tuple[lu.WrappedFun, Callable[[], Any]]:
    f = reap_function(f, trace.main, settings, True)
    return reap_wrapper(f, trace)


@lu.transformation_with_aux
def reap_wrapper(trace: HarvestTrace, *args):
    del trace
    out, reaps, metadata = yield (args,), {}
    out_flat, out_tree = jtu.tree_flatten((out, reaps))
    yield out_flat, (out_tree, metadata)


@lu.transformation
def reap_wrapper_drop_aux(trace: HarvestTrace, *args):
    del trace
    out, reaps, _ = yield (args,), {}
    out_flat, _ = jtu.tree_flatten((out, reaps))
    yield out_flat


@lu.transformation_with_aux
def _reap_metadata_wrapper(*args):
    out, reaps, metadata = yield (args,), {}
    yield (out, reaps), metadata


def _get_harvest_metadata(closed_jaxpr, settings, *args):
    """Probes a jaxpr for metadata like its sown values."""
    fun = lu.wrap_init(jc.jaxpr_as_fun(closed_jaxpr))
    with jc.new_main(HarvestTrace) as main:
        settings = HarvestSettings(settings.tag)
        fun = reap_function(fun, main, settings, True)
        fun, aux = _reap_metadata_wrapper(fun)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        in_avals = jax_util.safe_map(
            lambda a: jc.raise_to_shaped(jc.get_aval(a)), flat_args
        )
        pe.trace_to_jaxpr_final(flat_fun, in_avals)
        metadata = aux()
        out_tree()
    return metadata


def _reap_scan_rule(
    trace: HarvestTrace,
    *tracers,
    length,
    reverse,
    jaxpr,
    num_consts,
    num_carry,
    linear,
    unroll,
):
    """Reaps the body of a scan to pull out `clobber` and `append` sows."""

    const_tracers, carry_tracers, xs_tracers = jax_util.split_list(
        tracers, [num_consts, num_carry]
    )
    _, carry_avals, xs_avals = jtu.tree_map(
        lambda x: x.aval, (const_tracers, carry_tracers, xs_tracers)
    )
    const_vals, carry_vals, xs_vals = jtu.tree_map(
        lambda x: x.val, (const_tracers, carry_tracers, xs_tracers)
    )
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    x_tracers = [t[0] if hasattr(t, "_getitem") else t for t in xs_tracers]
    x_avals = [t.aval for t in x_tracers]
    x_vals = [t.val for t in x_tracers]
    metadata = _get_harvest_metadata(
        jaxpr, settings, *(const_vals + carry_vals + x_vals)
    )

    reap_modes = collections.defaultdict(set)
    reap_carry_avals = {}
    for name, meta in metadata.items():
        mode = meta["mode"]
        aval = meta["aval"]
        if mode == "strict":
            raise ValueError(f"Cannot use strict mode for '{name}' inside `scan`.")
        reap_modes[mode].add(name)
        if mode == "clobber":
            reap_carry_avals[name] = aval
    body_fun = jc.jaxpr_as_fun(jaxpr)

    reap_carry_flat_avals, _ = jtu.tree_flatten(reap_carry_avals)

    reap_carry_in_tree = jtu.tree_structure(((carry_avals, reap_carry_avals), xs_avals))

    def new_body(carry, x):
        carry, _ = carry
        all_values = const_vals + jtu.tree_leaves((carry, x))
        out, reaps = call_and_reap(
            body_fun,
            tag=settings.tag,
        )(*all_values)
        carry_out, y = jax_util.split_list(out, [num_carry])
        carry_reaps = {
            name: val for name, val in reaps.items() if name in reap_modes["clobber"]
        }
        xs_reaps = {
            name: val for name, val in reaps.items() if name in reap_modes["append"]
        }
        return (carry_out, carry_reaps), (y, xs_reaps)

    (new_body_jaxpr, consts, out_tree,) = lcf._initial_style_jaxpr(
        new_body,
        reap_carry_in_tree,
        tuple(carry_avals + reap_carry_flat_avals + x_avals),
    )
    dummy_reap_carry_vals = jtu.tree_map(
        lambda x: jnp.zeros(x.shape, x.dtype), reap_carry_flat_avals
    )
    out = lax.scan_p.bind(
        *(consts + carry_vals + dummy_reap_carry_vals + xs_vals),
        reverse=reverse,
        length=length,
        jaxpr=new_body_jaxpr,
        num_consts=len(consts),
        num_carry=len(carry_vals + dummy_reap_carry_vals),
        linear=(
            linear[: len(consts)]
            + (False,) * len(dummy_reap_carry_vals)
            + linear[len(consts) :]
        ),
        unroll=unroll,
    )
    (carry_out, carry_reaps), (ys, ys_reaps) = jtu.tree_unflatten(out_tree, out)
    (carry_out, carry_reaps), (ys, ys_reaps) = jtu.tree_map(
        trace.pure, ((carry_out, carry_reaps), (ys, ys_reaps))
    )
    for k, v in {**carry_reaps, **ys_reaps}.items():
        sow(v, tag=settings.tag, mode=metadata[k]["mode"], name=k)
    return carry_out + ys


reap_custom_rules[lcf.scan_p] = _reap_scan_rule


def _reap_while_rule(
    trace: HarvestTrace,
    *tracers,
    cond_jaxpr,
    body_jaxpr,
    cond_nconsts,
    body_nconsts,
):
    """Reaps the body of a while loop to get the reaps of the final
    iteration."""
    cond_const_tracers, body_const_tracers, init_tracers = jax_util.split_list(
        tracers, [cond_nconsts, body_nconsts]
    )
    _, init_avals = jtu.tree_map(lambda x: x.aval, (body_const_tracers, init_tracers))
    cond_const_vals, body_const_vals, init_vals = jtu.tree_map(
        lambda x: x.val, (cond_const_tracers, body_const_tracers, init_tracers)
    )
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    body_metadata = _get_harvest_metadata(
        body_jaxpr, settings, *(body_const_tracers + init_tracers)
    )
    for k, meta in body_metadata.items():
        mode = meta["mode"]
        if mode != "clobber":
            raise ValueError(
                f"Must use clobber mode for '{k}' inside of a `while_loop`."
            )
    reap_avals = {k: v["aval"] for k, v in body_metadata.items()}

    cond_fun = jc.jaxpr_as_fun(cond_jaxpr)
    body_fun = jc.jaxpr_as_fun(body_jaxpr)
    reap_settings = dict(
        tag=settings.tag,
    )

    def new_cond(carry, _):
        return cond_fun(*(cond_const_vals + carry))

    def new_body(carry, _):
        carry, reaps = call_and_reap(body_fun, **reap_settings)(
            *(body_const_vals + carry)
        )
        return (carry, reaps)

    new_in_avals, new_in_tree = jtu.tree_flatten((init_avals, reap_avals))
    (
        new_cond_jaxpr,
        cond_consts,
        _,
    ) = lcf._initial_style_jaxpr(new_cond, new_in_tree, tuple(new_in_avals))
    (
        new_body_jaxpr,
        body_consts,
        out_tree,
    ) = lcf._initial_style_jaxpr(new_body, new_in_tree, tuple(new_in_avals))
    dummy_reap_vals = jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), reap_avals)
    new_in_vals = jtu.tree_leaves((init_vals, dummy_reap_vals))
    out = lax.while_p.bind(
        *(cond_consts + body_consts + new_in_vals),
        cond_nconsts=len(cond_consts),
        body_nconsts=len(body_consts),
        cond_jaxpr=new_cond_jaxpr,
        body_jaxpr=new_body_jaxpr,
    )
    out = jax_util.safe_map(trace.pure, out)
    out, reaps = jtu.tree_unflatten(out_tree, out)
    for k, v in reaps.items():
        sow(v, name=k, tag=settings.tag, mode=body_metadata[k]["mode"])
    return out


reap_custom_rules[lcf.while_p] = _reap_while_rule


def _check_branch_metadata(branch_metadatas):
    """Checks that a set of harvest metadata are consistent with each other."""
    first_branch_meta = branch_metadatas[0]
    for branch_metadata in branch_metadatas[1:]:
        if len(branch_metadata) != len(first_branch_meta):
            raise ValueError("Mismatching number of `sow`s between branches.")
        for name, meta in branch_metadata.items():
            if name not in first_branch_meta:
                raise ValueError(f"Missing sow in branch: '{name}'.")
            first_meta_aval = first_branch_meta[name]["aval"]
            if meta["aval"].shape != first_meta_aval.shape:
                raise ValueError(f"Mismatched shape between branches: '{name}'.")
            if meta["aval"].dtype != first_meta_aval.dtype:
                raise ValueError(f"Mismatched dtype between branches: '{name}'.")


def _reap_cond_rule(trace, *tracers, branches, linear):
    """Reaps each path of the `cond`."""
    index_tracer, ops_tracers = tracers[0], tracers[1:]
    index_val, ops_vals = jtu.tree_map(lambda x: x.val, (index_tracer, ops_tracers))
    _, ops_avals = jtu.tree_map(lambda x: x.aval, (index_tracer, ops_tracers))
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    reap_settings = dict(tag=settings.tag)
    branch_metadatas = tuple(
        _get_harvest_metadata(branch, settings, *ops_tracers) for branch in branches
    )
    _check_branch_metadata(branch_metadatas)
    branch_funs = tuple(map(jc.jaxpr_as_fun, branches))
    reaped_branches = tuple(call_and_reap(f, **reap_settings) for f in branch_funs)
    in_tree = jtu.tree_structure(ops_avals)
    (
        new_branch_jaxprs,
        consts,
        out_trees,
    ) = lcf._initial_style_jaxprs_with_common_consts(
        reaped_branches, in_tree, ops_avals, lax.cond_p.name
    )
    out = lax.cond_p.bind(
        index_val,
        *(tuple(consts) + ops_vals),
        branches=tuple(new_branch_jaxprs),
        linear=(False,) * len(tuple(consts) + linear),
    )
    out = jax_util.safe_map(trace.pure, out)
    out, reaps = jtu.tree_unflatten(out_trees[0], out)
    for k, v in reaps.items():
        sow(v, name=k, tag=settings.tag, mode=branch_metadatas[0][k]["mode"])
    return out


reap_custom_rules[lcf.cond_p] = _reap_cond_rule


def _reap_checkpoint_rule(trace, *tracers, jaxpr, policy, prevent_cse, differentiated):
    """Reap checkpoint rule."""
    invals = [t.val for t in tracers]
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    reap_settings = dict(tag=settings.tag)
    closed_jaxpr = jc.ClosedJaxpr(jaxpr, ())
    reap_metadata = _get_harvest_metadata(closed_jaxpr, settings, *tracers)
    remat_fun = jc.jaxpr_as_fun(closed_jaxpr)
    reaped_remat_fun = call_and_reap(remat_fun, **reap_settings)
    (reap_jaxpr, consts, out_tree,) = lcf._initial_style_jaxpr(
        reaped_remat_fun,
        jtu.tree_structure(invals),
        tuple(t.aval for t in tracers),
    )
    outvals = ad_checkpoint.remat_p.bind(
        *consts,
        *invals,
        jaxpr=reap_jaxpr.jaxpr,
        policy=policy,
        prevent_cse=prevent_cse,
        differentiated=differentiated,
    )
    outvals = jax_util.safe_map(trace.pure, outvals)
    out, reaps = jtu.tree_unflatten(out_tree, outvals)
    for k, v in reaps.items():
        sow(v, name=k, tag=settings.tag, mode=reap_metadata[k]["mode"])
    return out


reap_custom_rules[ad_checkpoint.remat_p] = _reap_checkpoint_rule


@lu.cache
def _harvest_pjit_jaxpr(flat_fun, in_avals):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    if any(isinstance(c, jax_core.Tracer) for c in consts):
        jaxpr = pe.convert_constvars_jaxpr(jaxpr)
        jaxpr = pe.close_jaxpr(jaxpr)
        final_consts = consts
    else:
        jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
        final_consts = []

    return jaxpr, final_consts, out_avals


def _calc_extra_inps(num_consts, params):
    in_shardings = (sharding_impls.UNSPECIFIED,) * num_consts + params["in_shardings"]
    donated_invars = (False,) * num_consts + params["donated_invars"]
    return in_shardings, donated_invars


def _reap_pjit_rule(trace, *tracers, **params):
    """Reap pjit rule."""
    if params["in_shardings"] and not any(
        sharding_impls.is_unspecified(i) for i in params["in_shardings"]
    ):
        raise ValueError(
            "harvest only supports pjit which has no in_axis_resources "
            f'specified. Got {params["in_shardings"]}'
        )
    if params["out_shardings"] and not any(
        sharding_impls.is_unspecified(o) for o in params["out_shardings"]
    ):
        raise ValueError(
            "harvest only supports pjit which has no out_axis_resources "
            f'specified. Got {params["out_shardings"]}'
        )

    invals = [t.val for t in tracers]
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    reap_settings = dict(tag=settings.tag)
    closed_jaxpr = params["jaxpr"]
    reap_metadata = _get_harvest_metadata(closed_jaxpr, settings, *tracers)
    pjit_fun = jax_core.jaxpr_as_fun(closed_jaxpr)
    reaped_pjit_fun = lu.wrap_init(call_and_reap(pjit_fun, **reap_settings))
    in_tree = jtu.tree_structure(invals)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(reaped_pjit_fun, in_tree)

    reap_jaxpr, final_consts, out_avals = _harvest_pjit_jaxpr(
        flat_fun, tuple(t.aval for t in tracers)
    )
    in_shardings, donated_invars = _calc_extra_inps(len(final_consts), params)

    new_params = {
        **params,
        "jaxpr": reap_jaxpr,
        "out_shardings": (sharding_impls.UNSPECIFIED,) * len(out_avals),
        "in_shardings": in_shardings,
        "donated_invars": donated_invars,
    }
    outvals = pjit.pjit_p.bind(*final_consts, *invals, **new_params)

    outvals = jax_util.safe_map(trace.pure, outvals)
    out, reaps = jtu.tree_unflatten(out_tree(), outvals)
    for k, v in reaps.items():
        sow(v, name=k, tag=settings.tag, mode=reap_metadata[k]["mode"])
    return out


reap_custom_rules[pjit.pjit_p] = _reap_pjit_rule

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

    def get_custom_rule(self, primitive):
        return plant_custom_rules.get(primitive)

    def handle_sow(self, *values, meta, tag, tree, mode):
        """Returns the value stored in the plants dictionary."""
        if meta in self._already_planted and mode != "clobber":
            raise ValueError(f"Variable has already been planted: {meta}")
        if meta in self.plants:
            self._already_planted.add(meta)
            return jtu.tree_leaves(self.plants[meta])
        return sow_p.bind(*values, meta=meta, tag=tag, mode=mode, tree=tree)


@lu.transformation
def plant_function(
    main: jax_core.MainTrace,
    settings: HarvestSettings,
    in_tree: Any,
    args: Iterable[Any],
):
    """A function transformation that injects values in place of sows."""
    trace = HarvestTrace(main, jax_core.cur_sublevel())
    plants, args = jtu.tree_unflatten(in_tree, args)
    args = jax_util.safe_map(trace.pure, args)
    context = PlantContext(settings, plants)
    with staging.new_dynamic_context(main, context):
        ans = yield args, {}
        out_tracers = jax_util.safe_map(trace.full_raise, ans)
        del main
    yield [t.val for t in out_tracers]


def plant_eval(
    f: lu.WrappedFun, trace: HarvestTrace, settings: HarvestSettings, all_tree: Any
) -> Tuple[lu.WrappedFun, Callable[[], Any]]:
    f = plant_function(f, trace.main, settings, all_tree)
    return plant_wrapper(f)


@lu.transformation
def plant_wrapper(*args):
    out = yield (args,), {}
    yield out


def _plant_scan_rule(
    trace: HarvestTrace,
    *tracers,
    length,
    reverse,
    jaxpr,
    num_consts,
    num_carry,
    linear,
    unroll,
):
    """Injects values into a scan according to their sow mode."""

    const_tracers, carry_tracers, xs_tracers = jax_util.split_list(
        tracers, [num_consts, num_carry]
    )
    carry_avals, xs_avals = jtu.tree_map(lambda x: x.aval, (carry_tracers, xs_tracers))
    const_vals, carry_vals, xs_vals = jtu.tree_map(
        lambda x: x.val, (const_tracers, carry_tracers, xs_tracers)
    )
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    x_tracers = [t[0] if hasattr(t, "_getitem") else t for t in xs_tracers]
    x_avals = [t.aval for t in x_tracers]
    metadata = _get_harvest_metadata(
        jaxpr, settings, *(const_tracers + carry_tracers + x_tracers)
    )

    plants = context.plants
    plant_modes = collections.defaultdict(set)
    plant_xs_avals = {}
    for name, meta in metadata.items():
        mode = meta["mode"]
        aval = meta["aval"]
        if mode == "strict":
            raise ValueError(f"Cannot use strict mode for '{name}' inside `scan`.")
        plant_modes[mode].add(name)
        if mode == "append" and name in plants:
            plant_xs_avals[name] = aval
    body_fun = jax_core.jaxpr_as_fun(jaxpr)
    clobber_plants = {
        name: value for name, value in plants.items() if name in plant_modes["clobber"]
    }
    append_plants = {
        name: value for name, value in plants.items() if name in plant_modes["append"]
    }

    plant_xs_flat_avals, _ = jtu.tree_flatten(plant_xs_avals)

    plant_xs_in_tree = jtu.tree_structure((carry_avals, (xs_avals, plant_xs_avals)))

    def new_body(carry, x):
        x, plants = x
        all_plants = {**plants, **clobber_plants}
        all_values = const_vals + jtu.tree_leaves((carry, x))
        out = plant(body_fun, tag=settings.tag)(all_plants, *all_values)
        carry_out, y = jax_util.split_list(out, [num_carry])
        return carry_out, y

    (new_body_jaxpr, consts, _,) = lcf._initial_style_jaxpr(
        new_body, plant_xs_in_tree, tuple(carry_avals + x_avals + plant_xs_flat_avals)
    )
    plant_vals = jtu.tree_leaves(append_plants)
    out = lcf.scan_p.bind(
        *(consts + carry_vals + xs_vals + plant_vals),
        reverse=reverse,
        length=length,
        jaxpr=new_body_jaxpr,
        num_consts=len(consts),
        num_carry=num_carry,
        linear=linear + (False,) * len(plant_vals),
        unroll=unroll,
    )
    return out


plant_custom_rules[lcf.scan_p] = _plant_scan_rule


def _plant_while_rule(
    trace: HarvestTrace, *tracers, cond_jaxpr, body_jaxpr, cond_nconsts, body_nconsts
):
    """Injects values into a while loop, overriding values for all
    iterations."""
    cond_const_tracers, body_const_tracers, init_tracers = jax_util.split_list(
        tracers, [cond_nconsts, body_nconsts]
    )
    init_avals = jtu.tree_map(lambda x: x.aval, init_tracers)
    cond_const_vals, body_const_vals, init_vals = jtu.tree_map(
        lambda x: x.val, (cond_const_tracers, body_const_tracers, init_tracers)
    )
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    body_metadata = _get_harvest_metadata(
        body_jaxpr, settings, *(body_const_tracers + init_tracers)
    )
    for k, meta in body_metadata.items():
        mode = meta["mode"]
        if mode != "clobber":
            raise ValueError(
                f"Must use clobber mode for '{k}' inside of a `while_loop`."
            )

    body_fun = jax_core.jaxpr_as_fun(body_jaxpr)
    plant_settings = dict(tag=settings.tag)
    plants = context.plants

    def new_body(*carry):
        carry = plant(body_fun, **plant_settings)(
            plants, *(tuple(body_const_vals) + carry)
        )
        return carry

    in_tree = jtu.tree_structure(init_avals)
    (
        new_body_jaxpr,
        new_body_consts,
        _,
    ) = lcf._initial_style_jaxpr(new_body, in_tree, tuple(init_avals))
    out = lcf.while_p.bind(
        *(cond_const_vals + new_body_consts + init_vals),
        cond_nconsts=len(cond_const_vals),
        body_nconsts=len(new_body_consts),
        cond_jaxpr=cond_jaxpr,
        body_jaxpr=new_body_jaxpr,
    )
    return jax_util.safe_map(trace.pure, out)


plant_custom_rules[lcf.while_p] = _plant_while_rule


def _plant_cond_rule(trace, *tracers, branches, linear):
    """Injects the same values into both branches of a conditional."""
    index_tracer, ops_tracers = tracers[0], tracers[1:]
    index_val, ops_vals = jtu.tree_map(lambda x: x.val, (index_tracer, ops_tracers))
    ops_avals = jtu.tree_map(lambda x: x.aval, ops_tracers)
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    plant_settings = dict(tag=settings.tag)
    branch_metadatas = tuple(
        _get_harvest_metadata(branch, settings, *ops_tracers) for branch in branches
    )
    _check_branch_metadata(branch_metadatas)
    plants = context.plants
    branch_funs = tuple(map(jax_core.jaxpr_as_fun, branches))
    planted_branches = tuple(
        functools.partial(plant(f, **plant_settings), plants) for f in branch_funs
    )
    in_tree = jtu.tree_structure(ops_avals)
    (new_branch_jaxprs, consts, _,) = lcf._initial_style_jaxprs_with_common_consts(
        planted_branches, in_tree, ops_avals, lax.cond_p.name
    )
    out = lax.cond_p.bind(
        index_val,
        *(tuple(consts) + ops_vals),
        branches=tuple(new_branch_jaxprs),
        linear=(False,) * len(tuple(consts) + linear),
    )
    return jax_util.safe_map(trace.pure, out)


plant_custom_rules[lcf.cond_p] = _plant_cond_rule


def _plant_checkpoint_rule(trace, *tracers, jaxpr, policy, prevent_cse, differentiated):
    """Plant checkpoint rule."""
    invals = [t.val for t in tracers]
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    plant_settings = dict(tag=settings.tag)
    closed_jaxpr = jax_core.ClosedJaxpr(jaxpr, ())
    plants = context.plants
    remat_fun = jax_core.jaxpr_as_fun(closed_jaxpr)
    planted_remat_fun = functools.partial(plant(remat_fun, **plant_settings), plants)
    (plant_jaxpr, consts, _,) = lcf._initial_style_jaxpr(
        planted_remat_fun,
        jtu.tree_structure(invals),
        tuple(t.aval for t in tracers),
    )
    outvals = ad_checkpoint.remat_p.bind(
        *consts,
        *invals,
        jaxpr=plant_jaxpr.jaxpr,
        policy=policy,
        prevent_cse=prevent_cse,
        differentiated=differentiated,
    )
    return jax_util.safe_map(trace.pure, outvals)


plant_custom_rules[ad_checkpoint.remat_p] = _plant_checkpoint_rule


def _plant_pjit_rule(trace, *tracers, **params):
    """Plant pjit rule."""
    if params["in_shardings"] and not any(
        sharding_impls.is_unspecified(i) for i in params["in_shardings"]
    ):
        raise ValueError(
            "oryx only supports pjit which has no in_axis_resources "
            f'specified. Got {params["in_shardings"]}'
        )
    if params["out_shardings"] and not any(
        sharding_impls.is_unspecified(o) for o in params["out_shardings"]
    ):
        raise ValueError(
            "oryx only supports pjit which has no out_axis_resources "
            f'specified. Got {params["out_shardings"]}'
        )

    invals = [t.val for t in tracers]
    context = staging.get_dynamic_context(trace)
    settings = context.settings
    plant_settings = dict(tag=settings.tag)
    closed_jaxpr = params["jaxpr"]
    plants = context.plants

    pjit_fun = jax_core.jaxpr_as_fun(closed_jaxpr)
    planted_pjit_fun = lu.wrap_init(
        functools.partial(plant(pjit_fun, **plant_settings), plants)
    )
    in_tree = jtu.tree_structure(invals)
    flat_fun, _ = api_util.flatten_fun_nokwargs(planted_pjit_fun, in_tree)

    planted_jaxpr, final_consts, out_avals = _harvest_pjit_jaxpr(
        flat_fun, tuple(t.aval for t in tracers)
    )
    in_shardings, donated_invars = _calc_extra_inps(len(final_consts), params)

    new_params = {
        **params,
        "jaxpr": planted_jaxpr,
        "out_shardings": (sharding_impls.UNSPECIFIED,) * len(out_avals),
        "in_shardings": in_shardings,
        "donated_invars": donated_invars,
    }
    outvals = pjit.pjit_p.bind(*final_consts, *invals, **new_params)

    return jax_util.safe_map(trace.pure, outvals)


plant_custom_rules[pjit.pjit_p] = _plant_pjit_rule

##############
# Interfaces #
##############


def reap(
    fn,
    *,
    state: ReapState,
    tag: Hashable,
):
    settings = HarvestSettings(tag)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = ReapContext.new(settings, state)
        retvals, (reaps,) = context.transform(fn, ctx, HarvestTrace)(*args, **kwargs)
        return retvals, reaps

    return wrapper


def plant(
    fn,
    *,
    tag: Hashable,
):
    settings = HarvestSettings(tag)

    @functools.wraps(fn)
    def wrapper(plants, *args, **kwargs):
        ctx = PlantContext.new(settings, plants)
        retvals, _ = context.transform(fn, ctx, HarvestTrace)(*args, **kwargs)
        return retvals

    return wrapper


def harvest(
    fn: Callable,
    *,
    tag: Hashable,
):
    """> `harvest` is a function transformation that augments the behaviors of
    `sow`s in the function body. By default, invoking `sow` acts as the
    identity function and does not affect the semantics of a function.
    Harvesting `f` produces a function that can take advantage of `sow`s
    present in its execution.

    `harvest` is a function that takes in a function
    `f` and a `tag: String`. `harvest` will only interact with `sow`s whose tag
    matches the input `tag`. The returned function can interact with the `sow` invocations
    in the function body in either of two ways.

    * The first is via "injection",
    where intermediate values in the function values can be overridden.
    `harvest(f)` takes in an additional initial argument, `plants`, a
    dictionary mapping names to values. Each name in `plants` should correspond
    to a `sow` in `f`, and while running `harvest(f)` rather than using the
    value at runtime for the `sow`, we substitute in the value from the
    `plants` dictionary.

    * The other way in which `harvest(f)` interacts with
    `sow` invocations is that if it encounters a `sow` whose tag matches and whose name is
    *not* in `plants`, it will add the output of the `sow` to a dictionary
    mapping the sow name to its output, called `reaps`. The `reaps` dictionary,
    at the end of `harvest(f)`'s execution, will contain the outputs of all
    `sow`s whose values were not injected, or "planted.".

    The general convention is that, for any given execution of
    `harvest(f, tag=tag)`, there will be *no more remaining sow invocations of the
    given tag if the function were to be harvested again, i.e. if we were to
    nest harvests with the same tag `harvest(harvest(f, tag='some_tag'),
    tag='some_tag')`, the outer harvest would have nothing to plant or
    to reap.
    """

    @functools.wraps(fn)
    def wrapper(plants, *args, **kwargs):
        f = plant(fn, tag=tag)
        f = reap(f, tag=tag)
        return f(plants, *args, **kwargs)

    return wrapper
