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

from dataclasses import dataclass
from functools import partial, wraps

import jax.extend as jex
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import api_util
from jax import util as jax_util
from jax.core import eval_jaxpr
from jax.extend.core import Jaxpr
from jax.interpreters import batching
from jax.lax import broadcast, cond_p, scan, scan_p, switch

from genjax._src.core.compiler.initial_style_primitive import (
    ElaboratedPrimitive,
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.compiler.interpreters.environment import Environment
from genjax._src.core.compiler.staging import stage
from genjax._src.core.typing import Any, Array, Callable, PRNGKey, Sequence


def static_dim_length(in_axes, args: tuple[Any, ...]) -> int | None:
    # perform the in_axes massaging that vmap performs internally:
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)
    elif isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    def find_axis_size(axis: int | None, x: Any) -> int | None:
        """Find the size of the axis specified by `axis` for the argument `x`."""
        if axis is not None:
            leaf = jtu.tree_leaves(x)[0]
            return leaf.shape[axis]

    # tree_map uses in_axes as a template. To have passed vmap validation, Any non-None entry
    # must bottom out in an array-shaped leaf, and all such leafs must have the same size for
    # the specified dimension. Fetching the first is sufficient.
    axis_sizes = jtu.tree_leaves(
        jtu.tree_map(
            find_axis_size,
            in_axes,
            args,
            is_leaf=lambda x: x is None,
        )
    )
    return axis_sizes[0] if axis_sizes else None


######################
# Sampling primitive #
######################


class style:
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


sample_p = InitialStylePrimitive(
    f"{style.BOLD}{style.CYAN}pjax.sample{style.RESET}",
)
log_density_p = InitialStylePrimitive(
    f"{style.BOLD}{style.CYAN}pjax.log_density{style.RESET}",
)


# Zero-cost, just staging.
def make_flat(f):
    @wraps(f)
    def _make_flat(*args, **kwargs):
        debug_info = api_util.debug_info("_make_flat", f, args, kwargs)
        jaxpr, *_ = stage(f)(*args, **kwargs)

        def flat(*flat_args, **params):
            consts, args = jax_util.split_list(flat_args, [params["num_consts"]])
            return eval_jaxpr(jaxpr.jaxpr, consts, *args)

        return flat, debug_info

    return _make_flat


# This is very cheeky.
@dataclass
class GlobalKeyCounter:
    count: int = 0


# Very large source of unique keys.
global_counter = GlobalKeyCounter()

_fake_key = jrand.key(1)


def sample_binder(
    keyful_sampler: Callable[..., Any],
    name: str | None = None,
    batch_shape: tuple[int, ...] = (),
):
    keyful_with_batch_shape = partial(keyful_sampler, sample_shape=batch_shape)

    def sampler(*args, **kwargs):
        # We're playing a trick here by allowing users to invoke sample_p
        # without a key. So we hide it inside, and we pass this as the impl of `sample_p`.
        #
        # This is problematic for JIT, which will cache the statically generated key. But it's obvious to the user - their returned random choices won't change!
        #
        # The `seed` transformation below solves the JIT problem directly.
        def keyless(*args, **kwargs):
            global_counter.count += 1
            return keyful_with_batch_shape(
                jrand.key(global_counter.count),
                *args,
                **kwargs,
            )

        # Zero-cost, just staging.
        flat_keyful_sampler, _ = make_flat(keyful_with_batch_shape)(
            _fake_key, *args, **kwargs
        )

        # TODO: this shouldn't actually work.
        # Overload batching so that the primitive is retained
        # in the Jaxpr under vmap.
        # Holy smokes recursion.
        def batch(vector_args, batch_axes, **params):
            axis_size = params["axis_size"]
            n = static_dim_length(batch_axes, vector_args)
            outer_batch_dim = () if n is not None else (axis_size,) if axis_size else ()
            assert isinstance(outer_batch_dim, tuple)
            new_batch_shape = outer_batch_dim + batch_shape
            v = sample_binder(
                keyful_sampler,
                name=name,
                batch_shape=new_batch_shape,
            )(*vector_args)
            return (v,), (0 if n or axis_size else None,)

        return initial_style_bind(
            sample_p,
            keyful_sampler=keyful_sampler,
            flat_keyful_sampler=flat_keyful_sampler,
            batch=batch,
        )(keyless, dist=name)(*args, **kwargs)

    return sampler


def log_density_binder(
    log_density_impl: Callable[..., Any],
    name: str | None = None,
):
    def log_density(*args, **kwargs):
        # TODO: really not sure if this is right if you
        # nest vmaps...
        def batch(vector_args, batch_axes, **params):
            n = static_dim_length(batch_axes, tuple(vector_args))
            v = log_density_binder(log_density_impl, name=name)(*vector_args)
            return (v,), (0 if n else None,)

        return initial_style_bind(
            log_density_p,
            batch=batch,
        )(log_density_impl, dist=name)(*args, **kwargs)

    return log_density


####################
# Seed Interpreter #
####################


@dataclass
class SeedInterpreter:
    key: PRNGKey

    def eval_jaxpr_seed(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, args)
        for eqn in jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            primitive, inner_params = ElaboratedPrimitive.unwrap(eqn.primitive)
            if primitive == sample_p:
                invals = jax_util.safe_map(env.read, eqn.invars)
                args = subfuns + invals
                flat_keyful_sampler = inner_params["flat_keyful_sampler"]
                self.key, sub_key = jrand.split(self.key)
                outvals = flat_keyful_sampler(sub_key, *args, **inner_params)
            elif primitive == cond_p:
                invals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                branch_closed_jaxprs = params["branches"]
                self.key, sub_key = jrand.split(self.key)
                branches = tuple(
                    seed(jex.core.jaxpr_as_fun(branch))
                    for branch in branch_closed_jaxprs
                )
                index_val, ops_vals = invals[0], invals[1:]
                outvals = switch(
                    index_val,
                    branches,
                    sub_key,
                    *ops_vals,
                )

            # We replace the original scan with a new scan
            # that calls the interpreter on the scan body,
            # carries the key through and evolves it.
            elif primitive == scan_p:
                body_jaxpr = params["jaxpr"]
                length = params["length"]
                reverse = params["reverse"]
                num_consts = params["num_consts"]
                num_carry = params["num_carry"]
                const_vals, carry_vals, xs_vals = jax_util.split_list(
                    invals, [num_consts, num_carry]
                )

                body_fun = jex.core.jaxpr_as_fun(body_jaxpr)

                def new_body(carry, scanned_in):
                    (key, in_carry) = carry
                    (idx, in_scan) = scanned_in
                    all_values = const_vals + jtu.tree_leaves((in_carry, in_scan))
                    sub_key = jrand.fold_in(key, idx)
                    outs = seed(body_fun)(sub_key, *all_values)
                    out_carry, out_scan = jax_util.split_list(outs, [num_carry])
                    return (key, out_carry), out_scan

                self.key, sub_key = jrand.split(self.key)
                fold_idxs = jnp.arange(length)
                (_, flat_carry_out), scanned_out = scan(
                    new_body,
                    (sub_key, carry_vals),
                    (fold_idxs, xs_vals),
                    length=length,
                    reverse=reverse,
                )
                outvals = jtu.tree_leaves(
                    (flat_carry_out, scanned_out),
                )

            else:
                outvals = eqn.primitive.bind(*args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)

        return jax_util.safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_seed(
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def seed(
    f: Callable[..., Any],
):
    @wraps(f)
    def wrapped(key: PRNGKey, *args):
        interpreter = SeedInterpreter(key)
        return interpreter.run_interpreter(
            f,
            *args,
        )

    return wrapped


####################
# Vmap Interpreter #
####################


@dataclass
class BatchedArray:
    v: Array
    batched: int | None = None

    @classmethod
    def pure(cls, v):
        return BatchedArray(v, None)

    @classmethod
    def lift(cls, v):
        return BatchedArray(v[0], v[1])

    @classmethod
    def value(cls, v):
        return v.v if isinstance(v, BatchedArray) else v

    @classmethod
    def axis(cls, v):
        return v.batched if isinstance(v, BatchedArray) else None


def check_batched(vs: tuple[int | None, ...]):
    return any(map(lambda v: v == 0, vs))


@dataclass
class VmapInterpreter:
    in_axes: int | tuple[int | None, ...] | Sequence[Any] | None
    axis_size: int | None

    def eval_jaxpr_vmap(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        flat_args: list[Any],
        in_axes: list[int | None],
        axis_size: int | None,
    ):
        env = Environment()
        jax_util.safe_map(env.write, jaxpr.constvars, map(BatchedArray.pure, consts))
        jax_util.safe_map(
            env.write,
            jaxpr.invars,
            map(BatchedArray.lift, zip(flat_args, in_axes)),
        )
        for eqn in jaxpr.eqns:
            batched_invals = jax_util.safe_map(env.read, eqn.invars)
            batched_primals = list(map(BatchedArray.value, batched_invals))
            batched_axes = tuple(map(BatchedArray.axis, batched_invals))
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            vector_args = subfuns + batched_primals

            # Probabilistic.
            if ElaboratedPrimitive.check(eqn.primitive, sample_p):
                outvals, out_axes = batching.primitive_batchers[eqn.primitive](
                    tuple(vector_args),
                    batched_axes,
                    axis_size=axis_size,
                    **params,
                )

            # Deterministic and not control flow.
            else:
                if check_batched(batched_axes):
                    if eqn.primitive in batching.primitive_batchers:
                        outvals, out_axes = batching.primitive_batchers[eqn.primitive](
                            tuple(vector_args),
                            batched_axes,
                            **params,
                        )
                    else:
                        raise NotImplementedError()

                # Don't batch, just bind.
                else:
                    outvals = ElaboratedPrimitive.rebind(
                        eqn.primitive, *vector_args, **params
                    )
                    out_axes = (
                        None
                        if not eqn.primitive.multiple_results
                        else tuple(None for _ in outvals)
                    )

            if eqn.primitive.multiple_results:
                assert isinstance(out_axes, tuple)
                batched_outvals = list(map(BatchedArray.lift, zip(outvals, out_axes)))
            else:
                batched_outvals = [BatchedArray.lift((outvals, out_axes))]

            jax_util.safe_map(env.write, eqn.outvars, batched_outvals)

        batched_outvals = jax_util.safe_map(env.read, jaxpr.outvars)
        outvals = map(BatchedArray.value, batched_outvals)
        out_axes = map(BatchedArray.axis, batched_outvals)

        # Finally, broadcast any outval which wasn't batched during the process.
        if axis_size is not None:
            outvals = tuple(
                broadcast(v, (axis_size,)) if axis is None else v
                for (v, axis) in zip(outvals, out_axes)
            )
        return list(outvals)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        axis_size = (
            self.axis_size
            if self.axis_size
            else static_dim_length(self.in_axes, tuple(args))
        )
        if isinstance(self.in_axes, int | None):
            flat_axes = list((self.in_axes,) * len(flat_args))
        else:
            flat_axes, _ = jtu.tree_flatten(self.in_axes, is_leaf=lambda v: v is None)
        flat_out = self.eval_jaxpr_vmap(
            jaxpr,
            consts,
            flat_args,
            flat_axes,
            axis_size,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def vmap(
    f: Callable[..., Any],
    in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = 0,
    axis_size: int | None = None,
):
    """
    `pjax.vmap` extends `jax.vmap` to be "aware of probability distributions" as a first class citizen.
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        interpreter = VmapInterpreter(in_axes, axis_size)
        return interpreter.run_interpreter(f, *args, **kwargs)

    return wrapped
