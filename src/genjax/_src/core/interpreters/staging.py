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
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    BoolArray,
    Callable,
    static_check_is_concrete,
)

###############################
# Concrete Boolean arithmetic #
###############################


@Pytree.dataclass
class Flag(Pytree):
    """JAX compilation imposes restrictions on the control flow used in the compiled code.
    Branches gated by booleans must use GPU-compatible branching (e.g., `jax.lax.cond`).
    However, the GPU must compute both sides of the branch, wasting effort in the case
    where the gating boolean is constant. In such cases, if-based flow control will
    conceal the branch not taken from the JAX compiler, decreasing compilation time and
    code size for the result by not including the code for the branch that cannot be taken.

    This class contains a boolean value `f`, which is either native Python `True` or `False`,
    or a `jnp` array (typically of boolean dtype although this is not enforced either here
    or by JAX), together with a concreteness flag. Boolean operations are provided which
    preserve concreteness _when possible_ (i.e., admixture of a dynamic boolean with a concrete
    boolean may result in a dynamic boolean, if the value of the concrete boolean does not
    determine the result).
    """

    f: bool | BoolArray

    def and_(self, f: "Flag") -> "Flag":
        # True and X => X. False and X => False.
        if self.f is True:
            return f
        if self.f is False:
            return self
        if f.f is True:
            return self
        if f.f is False:
            return f
        return Flag(jnp.logical_and(self.f, f.f))

    def or_(self, f: "Flag") -> "Flag":
        # True or X => True. False or X => X.
        if self.f is True:
            return self
        if self.f is False:
            return f
        if f.f is True:
            return f
        if f.f is False:
            return self
        return Flag(jnp.logical_or(self.f, f.f))

    def not_(self) -> "Flag":
        if self.f is True:
            return Flag(False)
        elif self.f is False:
            return Flag(True)
        else:
            return Flag(jnp.logical_not(self.f))

    def concrete_true(self):
        return self.f is True

    def concrete_false(self):
        return self.f is False

    def __bool__(self) -> bool:
        return bool(jnp.all(self.f))

    def where(self, t: ArrayLike, f: ArrayLike) -> ArrayLike:
        """Return t or f according to the truth value contained in this flag
        in a manner that works in either the concrete or dynamic context"""
        if self.f is True:
            return t
        if self.f is False:
            return f
        return jax.lax.select(self.f, t, f)

    def cond(self, tf: Callable[..., Any], ff: Callable[..., Any], *args: Any):
        """Invokes `tf` with `args` if flag is true, else `ff`"""
        if self.f is True:
            return tf(*args)
        if self.f is False:
            return ff(*args)
        return jax.lax.cond(self.f, tf, ff, *args)

    @staticmethod
    def as_flag(f):
        if isinstance(f, Flag):
            return f
        return Flag(f)


def staged_check(v):
    return static_check_is_concrete(v) and v


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
        flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)  # pyright: ignore
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
