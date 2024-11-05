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


import typing
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from beartype.typing import overload
from jax import core as jc
from jax import tree_util as jtu
from jax.experimental import checkify
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.util import safe_map

from genjax._src.checkify import optional_check
from genjax._src.core.typing import (
    Any,
    Array,
    ArrayLike,
    Callable,
    Flag,
    Iterable,
    Sequence,
    TypeVar,
    static_check_is_concrete,
)

if TYPE_CHECKING:
    import genjax

WrappedFunWithAux = tuple[lu.WrappedFun, Callable[[], Any]]

###############################
# Concrete Boolean arithmetic #
###############################

R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


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
    def is_scalar(f: Flag) -> bool:
        """Check if a flag is scalar.

        A flag is considered scalar if it is either a Python bool or a JAX array with empty shape ().

        Args:
            f: The flag to check. Can be a Python bool or JAX array.

        Returns:
            bool: True if the flag is scalar, False otherwise.
        """
        return isinstance(f, bool) or f.shape == ()

    @staticmethod
    @overload
    def and_(f: bool, g: bool) -> bool: ...

    @staticmethod
    @overload
    def and_(f: Array, g: bool | Array) -> Array: ...

    @staticmethod
    @overload
    def and_(f: bool | Array, g: Array) -> Array: ...

    @staticmethod
    def and_(f: Flag, g: Flag) -> Flag:
        if isinstance(f, bool) and isinstance(g, bool):
            return f & g
        else:
            return jnp.logical_and(f, g)

    @staticmethod
    @overload
    def or_(f: bool, g: bool) -> bool: ...

    @staticmethod
    @overload
    def or_(f: Array, g: bool | Array) -> Array: ...

    @staticmethod
    @overload
    def or_(f: bool | Array, g: Array) -> Array: ...

    @staticmethod
    def or_(f: Flag, g: Flag) -> Flag:
        if isinstance(f, bool) and isinstance(g, bool):
            return f | g
        else:
            return jnp.logical_or(f, g)

    @staticmethod
    @overload
    def xor_(f: bool, g: bool) -> bool: ...

    @staticmethod
    @overload
    def xor_(f: Array, g: bool | Array) -> Array: ...

    @staticmethod
    @overload
    def xor_(f: bool | Array, g: Array) -> Array: ...

    @staticmethod
    def xor_(f: Flag, g: Flag) -> Flag:
        if isinstance(f, bool) and isinstance(g, bool):
            return f ^ g
        else:
            return jnp.logical_xor(f, g)

    @staticmethod
    @overload
    def not_(f: bool) -> bool: ...

    @staticmethod
    @overload
    def not_(f: Array) -> Array: ...

    @staticmethod
    def not_(f: Flag) -> Flag:
        match f:
            case True:
                return False
            case False:
                return True
            case _:
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


def tree_choose(
    idx: ArrayLike,
    pytrees: Sequence[R],
) -> R:
    """
    Version of `jax.numpy.choose` that

    - acts on lists of both `ArrayLike` and `Pytree` instances
    - acts like `vs[idx]` if `idx` is of type `int`.

    In the case of heterogenous types in `vs`, `tree_choose` will attempt to cast, or error if casting isn't possible. (mixed `bool` and `int` entries in `vs` will result in the cast of selected `bool` to `int`, for example.).

    Args:
        idx: The index used to select a value from `vs`.
        vs: A list of `Pytree` or `ArrayLike` values to choose from.

    Returns:
        The selected value from the list.
    """

    def inner(*vs: ArrayLike) -> ArrayLike:
        # Computing `result` above the branch allows us to:
        # - catch incompatible types / shapes in the result
        # - in the case of compatible types requiring casts (like bool => int),
        #   result's dtype tells us the final type.
        result = jnp.choose(idx, vs, mode="wrap")
        if isinstance(idx, int):
            return jnp.asarray(vs[idx % len(vs)], dtype=result.dtype)
        else:
            return result

    return jtu.tree_map(inner, *pytrees)


def multi_switch(
    idx, branches: Iterable[Callable[..., Any]], arg_tuples: Iterable[tuple[Any, ...]]
):
    """
    A wrapper around switch that allows selection between functions with differently-shaped return values.

    This function enables switching between branches that may have different output shapes.
    It creates a list of placeholder shapes for each branch and then uses a switch statement
    to select the appropriate function to fill in the correct shape.

    Args:
        idx: The index used to select the branch. If the index is out of bounds, it will be clamped to within bounds.
        branches: An iterable of callable functions representing different branches.
        arg_tuples: An iterable of argument tuples, one for each branch function.

    Returns:
        The result of calling the selected branch function with its corresponding arguments.

    Note:
        This function assumes that the number of branches matches the number of argument tuples.
        Each branch function should be able to handle its corresponding argument tuple.
    """

    def _make_setter(static_idx: int, f: Callable[..., Any], args: tuple[Any, ...]):
        def set_result(shapes: list[R]) -> list[R]:
            shapes[static_idx] = f(*args)
            return shapes

        return set_result

    pairs = list(zip(branches, arg_tuples))
    shapes = list(to_shape_fn(f, jnp.zeros)(*args) for f, args in pairs)
    fns = list(_make_setter(i, f, args) for i, (f, args) in enumerate(pairs))
    return jax.lax.switch(idx, fns, operand=shapes)


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


def to_shape_fn(
    callable: F, fill_fn: Callable[[tuple[int], jnp.dtype[Any]], Array] | None = None
) -> F:
    """
    Convert a callable to a function that returns an empty pytree with the same structure as the original output (without any FLOPs).

    This function is similar to `jax.eval_shape`, but allows for optional post-processing of the output tree.

    Args:
        callable: The function to convert.
        fill_fn: A function to fill the output shapes with values. If None, returns the empty pytree as-is.
            The fill function takes a shape tuple and dtype as input and should return an array.

    Returns:
        A wrapped function that returns an empty pytree or a filled pytree with the same structure as the original function's output.
    """

    def wrapped(*args, **kwargs):
        shape = jax.eval_shape(callable, *args, **kwargs)
        if fill_fn is not None:
            f = fill_fn
            return jtu.tree_map(lambda x: f(x.shape, x.dtype), shape)
        else:
            return shape

    return typing.cast(F, wrapped)


_fake_key = jnp.array([0, 0], dtype=jnp.uint32)


def empty_trace(
    gen_fn: "genjax.GenerativeFunction[R]", args: "genjax.Arguments"
) -> "genjax.Trace[R]":
    """
    Create an empty trace for a generative function with given arguments (without spending any FLOPs).

    This function returns a trace with the same structure as a real trace, but filled with zero values. This is useful for static analysis and shape inference.

    Args:
        gen_fn: The generative function.
        args: The arguments to the generative function.

    Returns:
        A trace with the same structure as a real trace, but filled with zero values.
    """
    return to_shape_fn(gen_fn.simulate, jnp.zeros)(_fake_key, args)
