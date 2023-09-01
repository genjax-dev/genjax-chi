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

from dataclasses import dataclass

import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import primitives
from genjax._src.core.interpreters.staging import is_concrete
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import Optional
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


##############
# Primitives #
##############

# Generative function trace intrinsic.
trace_p = primitives.InitialStylePrimitive("trace")

# Cache intrinsic.
cache_p = primitives.InitialStylePrimitive("cache")


#####
# Static address checks
#####


# Usage in intrinsics: ensure that addresses do not contain JAX traced values.
def static_check_address_type(addr):
    check = all(jtu.tree_leaves(jtu.tree_map(is_concrete, addr)))
    if not check:
        raise Exception("Addresses must not contained JAX traced values.")


# Usage in intrinsics: ensure that addresses do not contain JAX traced values.
def static_check_concrete_or_dynamic_int_address(addr):
    def _check(v):
        if is_concrete(v):
            return True
        else:
            # TODO: fix to be more robust to different bit types.
            return v.dtype == jnp.int32

    check = all(jtu.tree_leaves(jtu.tree_map(_check, addr)))
    if not check:
        raise Exception(
            "Addresses must contain concrete (non-traced) values or traced integer values."
        )


#####
# Abstract generative function call
#####


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(gen_fn, _, *args):
    return gen_fn.__abstract_call__(*args)


############################################################
# Trace call (denotes invocation of a generative function) #
############################################################


@dataclass
class PytreeAddress(Pytree):
    static_rest: Tuple
    optional_leading_int_arr: Optional[IntArray]

    def flatten(self):
        if self.optional_leading_int_arr is None:
            return (), (self.static_rest, self.optional_leading_int_arr)
        else:
            return (self.optional_leading_int_arr,), (self.static_rest,)

    @classmethod
    @dispatch
    def new(cls, addr: Tuple):
        fst, *rst = addr
        if is_concrete(fst):
            return PytreeAddress((fst, *rst), None)
        else:
            return PytreeAddress(tuple(rst), fst)

    @classmethod
    @dispatch
    def new(cls, addr: IntArray):
        return PytreeAddress((), addr)

    @classmethod
    @dispatch
    def new(cls, addr: Any):
        return PytreeAddress((addr,), None)

    def to_tuple(self):
        if self.optional_leading_int_arr is None:
            return self.static_rest
        else:
            return (self.optional_leading_int_arr, *self.static_rest)


pytree_address = PytreeAddress.new


def _trace(gen_fn, addr, *args, **kwargs):
    return primitives.initial_style_bind(trace_p)(_abstract_gen_fn_call)(
        gen_fn, addr, *args, **kwargs
    )


@typecheck
def trace(addr: Any, gen_fn: GenerativeFunction, **kwargs) -> Callable:
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of `BuiltinGenerativeFunction`.

    Returns:
        callable: A callable which wraps the `trace_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `BuiltinGenerativeFunction` transformations.
    """
    assert isinstance(gen_fn, GenerativeFunction)
    static_check_concrete_or_dynamic_int_address(addr)
    pytree_addr = pytree_address(addr)
    return lambda *args: _trace(gen_fn, pytree_addr, *args, **kwargs)


##############################################################
# Caching (denotes caching of deterministic subcomputations) #
##############################################################


def _cache(fn, addr, *args, **kwargs):
    return primitives.initial_style_bind(cache_p)(fn)(fn, *args, addr, **kwargs)


@typecheck
def cache(addr: Any, fn: Callable, *args: Any, **kwargs) -> Callable:
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a function invocation.
        fn: A deterministic function whose return value is cached under the arguments (memoization) inside `BuiltinGenerativeFunction` traces.

    Returns:
        callable: A callable which wraps the `cache_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `BuiltinGenerativeFunction` transformations.
    """
    # fn must be deterministic.
    assert not isinstance(fn, GenerativeFunction)
    static_check_address_type(addr)
    return lambda *args: _cache(fn, addr, *args, **kwargs)
