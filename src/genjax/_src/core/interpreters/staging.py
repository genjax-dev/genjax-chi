# Copyright 2024 The MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
import jax.numpy as jnp
from jax import core as jc
from jax import make_jaxpr
from jax import tree_util as jtu
from jax.experimental import checkify
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.util import safe_map

from genjax._src.checkify import optional_check
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    Flag,
    TypeVar,
    static_check_is_concrete,
)

WrappedFunWithAux = tuple[lu.WrappedFun, Callable[[], Any]]

###############################
# Concrete Boolean arithmetic #
###############################

R = TypeVar("R")


class FlagOp:
    """JAX compilation imposes restrictions on the control flow used in the compiled code.
    Branches gated by booleans must use GPU-compatible branching (e.g., `jax.lax.cond`).
    However, the GPU must compute both sides of the branch, wasting effort in the case
    where the gating boolean is constant. In such cases, if-based flow control will
    conceal the branch not taken from the JAX compiler, decreasing compilation time and
    code size for the result by not including the code for the branch that cannot be taken.

    This class centralizes the concrete short-cut logic used by GenJAX.
    """

    @staticmethod
    def and_(f: Flag, g: Flag) -> Flag:
        # True and X => X. False and X => False.
        if f is True:
            return g
        if f is False:
            return f
        if g is True:
            return f
        if g is False:
            return g
        return jnp.logical_and(f, g)

    @staticmethod
    def or_(f: Flag, g: Flag) -> Flag:
        # True or X => True. False or X => X.
        if f is True:
            return f
        if f is False:
            return g
        if g is True:
            return g
        if g is False:
            return f
        return jnp.logical_or(f, g)

    @staticmethod
    def xor_(f: Flag, g: Flag) -> Flag:
        # True xor X => ~X. False xor X => X.
        if f is True:
            return FlagOp.not_(g)
        if f is False:
            return g
        if g is True:
            return FlagOp.not_(f)
        if g is False:
            return f
        return jnp.logical_xor(f, g)

    @staticmethod
    def not_(f: Flag) -> Flag:
        if f is True:
            return False
        if f is False:
            return True
        return jnp.logical_not(f)

    @staticmethod
    def concrete_true(f: Flag) -> bool:
        return f is True

    @staticmethod
    def concrete_false(f: Flag) -> bool:
        return f is False

    @staticmethod
    def where(f: Flag, tf: ArrayLike, ff: ArrayLike) -> ArrayLike:
        """Return tf or ff according to the truth value contained in flag
        in a manner that works in either the concrete or dynamic context"""
        if f is True:
            return tf
        if f is False:
            return ff
        return jax.lax.select(f, tf, ff)

    @staticmethod
    def cond(f: Flag, tf: Callable[..., R], ff: Callable[..., R], *args: Any) -> R:
        """Invokes `tf` with `args` if flag is true, else `ff`"""
        if f is True:
            return tf(*args)
        if f is False:
            return ff(*args)
        return jax.lax.cond(f, tf, ff, *args)


def staged_check(v):
    return static_check_is_concrete(v) and v


#########################
# Staged error handling #
#########################


def staged_err(check: Flag, msg, **kwargs):
    if FlagOp.concrete_true(check):
        raise Exception(msg)
    elif FlagOp.concrete_false(check):
        pass
    else:

        def _check():
            checkify.check(check, msg, **kwargs)

        optional_check(_check)


#######################################
# Staging utilities for type analysis #
#######################################


def get_shaped_aval(x):
    return jc.raise_to_shaped(jc.get_aval(x))


@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = jc.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


@lu.transformation_with_aux
def _flatten_fun_nokwargs(in_tree, *args_flat):
    py_args = jtu.tree_unflatten(in_tree, args_flat)
    ans = yield py_args, {}
    yield jtu.tree_flatten(ans)


# Wrapper to assign a correct type.
flatten_fun_nokwargs: Callable[[lu.WrappedFun, Any], WrappedFunWithAux] = (
    _flatten_fun_nokwargs  # pyright: ignore[reportAssignmentType]
)


def stage(f):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        typed_jaxpr = cached_stage_dynamic(flat_fun, tuple(flat_avals))
        return typed_jaxpr, (flat_args, in_tree, out_tree)

    return wrapped


def get_data_shape(callable):
    """
    Returns a function that stages a function and returns the abstract
    Pytree shapes of its return value.
    """

    def wrapped(*args):
        _, data_shape = make_jaxpr(callable, return_shape=True)(*args)
        return data_shape

    return wrapped


def get_trace_shape(gen_fn, args):
    key = jax.random.PRNGKey(0)
    return get_data_shape(gen_fn.simulate)(key, args)


def get_importance_shape(gen_fn, constraint, args):
    key = jax.random.PRNGKey(0)
    return get_data_shape(gen_fn.importance)(key, constraint, args)


def get_update_shape(gen_fn, tr, problem):
    key = jax.random.PRNGKey(0)
    return get_data_shape(gen_fn.update)(key, tr, problem)
