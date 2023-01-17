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

import jax.core as core
import jax.tree_util as jtu

from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.staging import stage


##############
# Primitives #
##############

# Generative function trace intrinsic.
gen_fn_p = core.Primitive("trace")

# Cache intrinsic.
cache_p = core.Primitive("cache")

############################################################
# Trace call (denotes invocation of a generative function) #
############################################################


def _trace(gen_fn, addr, *args, **kwargs):
    assert isinstance(gen_fn, GenerativeFunction)
    flat_args, tree_in = jtu.tree_flatten((gen_fn, args))
    retvals = gen_fn_p.bind(
        *flat_args,
        addr=addr,
        tree_in=tree_in,
        **kwargs,
    )
    # Collapse single arity returns.
    retvals = tuple(retvals)
    if len(retvals) == 1:
        retvals = retvals[0]
    return retvals


def trace(addr, call, **kwargs):
    return lambda *args: _trace(
        call,
        addr,
        *args,
        **kwargs,
    )


#####
# Abstract evaluation for trace
#####

# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(gen_fn, *args):
    return gen_fn.__abstract_call__(*args)


def gen_fn_abstract_eval(*args, addr, tree_in, **kwargs):
    gen_fn, args = jtu.tree_unflatten(tree_in, args)

    # See note above on `_abstract_gen_fn_call`.
    closed_jaxpr, _ = stage(_abstract_gen_fn_call)(gen_fn, *args)

    retvals = closed_jaxpr.out_avals[1:]
    return retvals


gen_fn_p.def_abstract_eval(gen_fn_abstract_eval)
gen_fn_p.multiple_results = True
gen_fn_p.must_handle = True


##############################################################
# Caching (denotes caching of deterministic subcomputations) #
##############################################################


def cache(addr, call, *args, **kwargs):
    assert isinstance(args, tuple)
    assert not isinstance(call, GenerativeFunction)
    return cache_p.bind(
        *args,
        addr=addr,
        fn=call,
        **kwargs,
    )


#####
# Abstract evaluation for cache
#####


def cache_abstract_eval(*args, addr, fn, **kwargs):
    closed_jaxpr, _ = stage(fn)(*args)
    return closed_jaxpr.out_avals


cache_p.def_abstract_eval(cache_abstract_eval)
cache_p.multiple_results = False
cache_p.must_handle = True
