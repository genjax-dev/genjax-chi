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

from jax import make_jaxpr
import jax.core as core
from genjax.core.datatypes import GenerativeFunction

#####
# Primitives
#####

# Generative function trace intrinsic_gen_fn.
gen_fn_p = core.Primitive("trace")

#####
# trace
#####


def _trace(addr, call, key, args, **kwargs):
    assert isinstance(args, tuple)
    if not isinstance(call, GenerativeFunction):
        raise Exception(
            "`trace` must have an instance of `GenerativeFunction`, not a `Callable`"
        )
    else:
        return gen_fn_p.bind(
            key,
            *args,
            addr=addr,
            gen_fn=call,
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


def gen_fn_abstract_eval(key, *args, addr, gen_fn, **kwargs):
    jaxpr = make_jaxpr(gen_fn.__call__)(key, *args)
    return jaxpr.out_avals


gen_fn_p.def_abstract_eval(gen_fn_abstract_eval)
gen_fn_p.multiple_results = True
gen_fn_p.must_handle = True
