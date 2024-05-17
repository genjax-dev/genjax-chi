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
from jax import api_util, make_jaxpr
from jax import core as jc
from jax import tree_util as jtu
from jax.experimental import checkify
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.lax import cond
from jax.util import safe_map

from genjax._src.checkify import optional_check
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import Bool, Int, static_check_is_concrete

register_exclusion(__file__)

###############################
# Concrete Boolean arithmetic #
###############################


def staged_check(v):
    return static_check_is_concrete(v) and v


def staged_and(x, y):
    if (
        static_check_is_concrete(x)
        and static_check_is_concrete(y)
        and isinstance(x, Bool)
        and isinstance(y, Bool)
    ):
        return x and y
    else:
        return jnp.logical_and(x, y)


def staged_or(x, y):
    # Static scalar land.
    if (
        static_check_is_concrete(x)
        and static_check_is_concrete(y)
        and isinstance(x, Bool)
        and isinstance(y, Bool)
    ):
        return x or y
    # Array land.
    else:
        return jnp.logical_or(x, y)


def staged_not(x):
    if static_check_is_concrete(x) and isinstance(x, Bool):
        return not x
    else:
        return jnp.logical_not(x)


def staged_switch(idx, v1, v2):
    if static_check_is_concrete(idx) and isinstance(idx, Int):
        return [v1, v2][idx]
    else:
        return cond(idx, lambda: v1, lambda: v2)


#########################
# Staged error handling #
#########################


def staged_err(check, msg, **kwargs):
    if static_check_is_concrete(check) and isinstance(check, Bool):
        if check:
            raise Exception(msg)
        else:
            return None
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


def stage(f):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
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


def get_update_shape(gen_fn, tr, problem, argdiffs):
    key = jax.random.PRNGKey(0)
    return get_data_shape(gen_fn.update)(key, tr, problem, argdiffs)


def make_zero_trace(gen_fn, *args):
    out_tree = get_trace_shape(gen_fn, *args)
    return jtu.tree_map(
        lambda v: jnp.zeros(v.shape, v.dtype),
        out_tree,
    )
