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
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    BoolArray,
    Int,
    static_check_is_concrete,
)

register_exclusion(__file__)

###############################
# Concrete Boolean arithmetic #
###############################


def flag(b: bool | BoolArray):
    return Flag(b, not isinstance(b, jc.Tracer))


@Pytree.dataclass(eq=False)
class Flag(Pytree):
    f: bool | BoolArray
    concrete: bool = Pytree.static()

    def and_(self, f: "Flag") -> "Flag":
        if self.concrete and f.concrete:
            return Flag(self.f and f.f, True)
        else:
            return Flag(jnp.logical_and(self.f, f.f), False)

    def or_(self, f: "Flag") -> "Flag":
        if self.concrete and f.concrete:
            return Flag(self.f or f.f, True)
        else:
            return Flag(jnp.logical_or(self.f, f.f), False)

    def not_(self) -> "Flag":
        if self.concrete:
            return Flag(not self.f, True)
        else:
            return Flag(jnp.logical_not(self.f), False)

    def concrete_true(self):
        return self.concrete and self.f

    def concrete_false(self):
        return self.concrete and not self.f

    def __eq__(self, other) -> bool:
        if not isinstance(other, Flag):
            return False
        return bool(self) == bool(other)

    def __bool__(self) -> bool:
        if self.concrete:
            return bool(self.f)
        else:
            return bool(jnp.all(self.f))


def staged_check(v):
    return static_check_is_concrete(v) and v


def staged_and(x, y) -> BoolArray:
    with jax.ensure_compile_time_eval():
        return jnp.logical_and(x, y)


def staged_or(x, y) -> BoolArray:
    with jax.ensure_compile_time_eval():
        return jnp.logical_or(x, y)


def staged_not(x) -> BoolArray:
    with jax.ensure_compile_time_eval():
        return jnp.logical_not(x)


def staged_switch(idx, v1, v2):
    if static_check_is_concrete(idx) and isinstance(idx, Int):
        return [v1, v2][idx]
    else:
        return jax.lax.cond(idx, lambda: v1, lambda: v2)


#########################
# Staged error handling #
#########################


def staged_err(check: Flag, msg, **kwargs):
    if check.concrete_true():
        raise Exception(msg)
    elif check.concrete_false():
        pass
    else:

        def _check():
            checkify.check(check.f, msg, **kwargs)

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


# This function has been cloned from api_util, since it is not exported from that module
@lu.transformation_with_aux
def flatten_fun_nokwargs(in_tree, *args_flat):
    py_args = jtu.tree_unflatten(in_tree, args_flat)
    ans = yield py_args, {}
    yield jtu.tree_flatten(ans)


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


def make_zero_trace(gen_fn, *args):
    out_tree = get_trace_shape(gen_fn, *args)
    return jtu.tree_map(
        lambda v: jnp.zeros(v.shape, v.dtype),
        out_tree,
    )
