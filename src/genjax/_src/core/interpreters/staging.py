# Copyright 2022 The MIT Probabilistic Computing Project
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

import contextlib
import operator
import threading
from functools import reduce

import jax
import jax.numpy as jnp
from jax import api_util
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util as jtu
from jax._src import dtypes
from jax.interpreters import partial_eval as pe
from jax.random import KeyArray

from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Generator
from genjax._src.core.typing import List


safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip


def get_shaped_aval(x):
    """Converts a JAX value type into a shaped abstract value."""
    # TODO: This is a kludge. Abstract evaluation currently breaks
    # on `random_wrap` without this branch.
    if isinstance(x, KeyArray):
        return jax_core.raise_to_shaped(jax_core.get_aval(x))

    if hasattr(x, "dtype") and hasattr(x, "shape"):
        return jax_core.ShapedArray(x.shape, dtypes.canonicalize_dtype(x.dtype))
    return jax_core.raise_to_shaped(jax_core.get_aval(x))


def pv_like(x, abstract=True):
    """Converts a JAX value type into a JAX `PartialVal`."""
    if abstract:
        return pe.PartialVal.unknown(get_shaped_aval(x))
    else:
        return pe.PartialVal((None, x))  # pytype: disable=wrong-arg-types


@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


@lu.cache
def cached_stage_partial(flat_fun, in_avals):
    pvals = [pe.PartialVal.unknown(aval) for aval in in_avals]
    jaxpr, _, consts = pe.trace_to_jaxpr(flat_fun, pvals, instantiate=True)
    typed_jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


def stage(f, dynamic=True):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        if dynamic:
            typed_jaxpr = cached_stage_dynamic(flat_fun, tuple(flat_avals))
        else:
            typed_jaxpr = cached_stage_partial(flat_fun, tuple(flat_avals))
        return typed_jaxpr, (flat_args, in_tree, out_tree())

    return wrapped


def trees(f):
    """Returns a function that determines input and output pytrees from inputs,
    and also returns the flattened input arguments."""

    def wrapped(*args, **kwargs):
        return stage(f)(*args, **kwargs)[1]

    return wrapped


class _ThreadLocalState(threading.local):
    def __init__(self):
        super().__init__()
        self.dynamic_contexts: Dict[jax_core.MainTrace, List[Any]] = {}


_thread_local_state = _ThreadLocalState()


@contextlib.contextmanager
def new_dynamic_context(
    master: jax_core.MainTrace, context: Any
) -> Generator[None, None, None]:
    """Creates a dynamic context for a trace."""
    if master not in _thread_local_state.dynamic_contexts:
        _thread_local_state.dynamic_contexts[master] = []
    _thread_local_state.dynamic_contexts[master].append(context)
    try:
        yield
    finally:
        _thread_local_state.dynamic_contexts[master].pop()
        if not _thread_local_state.dynamic_contexts[master]:
            del _thread_local_state.dynamic_contexts[master]


def get_dynamic_context(trace: jax_core.Trace) -> Any:
    """Returns the current active dynamic context for a trace."""
    if trace.main not in _thread_local_state.dynamic_contexts:
        raise ValueError(f"No dynamic context registered for trace: {trace}")
    return _thread_local_state.dynamic_contexts[trace.main][-1]


def extract_call_jaxpr(primitive, params):
    if not (primitive.call_primitive or primitive.map_primitive):
        return None, params
    else:
        params = dict(params)
        return params.pop("call_jaxpr"), params


#####
# Concretization
#####

# Force evaluation at compile time, if possible.
#
# These utilities are used throughout the codebase -- to gain some
# confidence that tracing will actually collapse potential branches when
# values are known statically.


def is_concrete(x):
    return not isinstance(x, jax.core.Tracer)


def concrete_and(*args):
    # First, walk through arguments and, if any are concrete
    # False, just return False.
    for k in args:
        if is_concrete(k) and not k:
            return False
    # Now, reduce over arguments. If all are concrete True,
    # return True.
    if all(map(is_concrete, args)):
        return reduce(operator.and_, args, True)
    # Else, return a dynamic Bool value.
    else:
        return jnp.logical_and(*args)


def concrete_cond(pred, true_branch, false_branch, *args):
    if is_concrete(pred):
        if pred:
            return true_branch(*args)
        else:
            return false_branch(*args)
    else:
        return jax.lax.cond(pred, true_branch, false_branch, *args)


def concrete_switch(index, branches, *args):
    if is_concrete(index):
        return branches[index](*args)
    else:
        return jax.lax.switch(index, branches, *args)
