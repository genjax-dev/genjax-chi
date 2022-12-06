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

from jax import abstract_arrays
from jax import api_util
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util as jtu
from jax._src import dtypes
from jax.interpreters import partial_eval as pe
from jax.random import KeyArray


safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip


def get_shaped_aval(x):
    """Converts a JAX value type into a shaped abstract value."""

    # TODO: This is a kludge. Abstract evaluation currently breaks
    # on `random_wrap` without this branch.
    if isinstance(x, KeyArray):
        return abstract_arrays.raise_to_shaped(jax_core.get_aval(x))

    if hasattr(x, "dtype") and hasattr(x, "shape"):
        return abstract_arrays.ShapedArray(
            x.shape, dtypes.canonicalize_dtype(x.dtype)
        )
    return abstract_arrays.raise_to_shaped(jax_core.get_aval(x))


def pv_like(x, abstract=True):
    """Converts a JAX value type into a JAX `PartialVal`."""
    if abstract:
        return pe.PartialVal.unknown(get_shaped_aval(x))
    else:
        return pe.PartialVal((None, x))  # pytype: disable=wrong-arg-types


def stage(f, dynamic=True):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        if dynamic:
            jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
        else:
            pvals = [pe.PartialVal.unknown(aval) for aval in flat_avals]
            jaxpr, _, consts = pe.trace_to_jaxpr(
                flat_fun, pvals, instantiate=True
            )
        typed_jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
        return typed_jaxpr, (in_tree, out_tree())

    return wrapped


def trees(f):
    """Returns a function that determines input and output pytrees from
    inputs."""

    def wrapped(*args, **kwargs):
        return stage(f)(*args, **kwargs)[1]

    return wrapped


def extract_call_jaxpr(primitive, params):
    if not (primitive.call_primitive or primitive.map_primitive):
        return None, params
    else:
        params = dict(params)
        return params.pop("call_jaxpr"), params
