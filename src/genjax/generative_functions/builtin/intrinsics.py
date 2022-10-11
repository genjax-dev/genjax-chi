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
from jax import make_jaxpr

from genjax.core.datatypes import GenerativeFunction


#####
# Primitives
#####

# Generative function trace intrinsic.
gen_fn_p = core.Primitive("trace")

#####
# trace
#####


def _trace(addr, call, key, args, **kwargs):
    assert isinstance(args, tuple)
    assert isinstance(call, GenerativeFunction)
    args, args_form = jtu.tree_flatten(args)
    return gen_fn_p.bind(
        key,
        *args,
        addr=addr,
        gen_fn=call,
        args_form=args_form,
        **kwargs,
    )


def trace(addr, call, **kwargs):
    return lambda key, args: _trace(
        addr,
        call,
        key,
        args,
        **kwargs,
    )


#####
# intrinsic_gen_fn
#####


def gen_fn_abstract_eval(key, *args, addr, gen_fn, args_form, **kwargs):
    args = jtu.tree_unflatten(args_form, args)

    # TODO: make sure this works in general.
    def _inner(key, *args):
        return gen_fn.__call__(key, *args, **kwargs)

    jaxpr = make_jaxpr(_inner)(key, *args)
    return jaxpr.out_avals


gen_fn_p.def_abstract_eval(gen_fn_abstract_eval)
gen_fn_p.multiple_results = True
gen_fn_p.must_handle = True
