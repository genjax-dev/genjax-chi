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
from jax._src import abstract_arrays
import inspect
from genjax.encapsulated import EncapsulatedGenerativeFunction

#####
# Primitives
#####

# GenJAX language trace primitive.
trace_p = core.Primitive("trace")

# External generative function trace primitive.
encapsulated_p = core.Primitive("encapsulated")

# Hierarchical (call) addressing primitive.
splice_p = core.Primitive("splice")

# Hierarchical (call) addressing primitive.
unsplice_p = core.Primitive("unsplice")

#####
# trace
#####


def _trace(addr, prim, *args, **kwargs):
    if inspect.isclass(prim):
        return trace_p.bind(*args, addr=addr, prim=prim, **kwargs)
    elif isinstance(prim, EncapsulatedGenerativeFunction):
        return encapsulated_p.bind(*args, addr=addr, prim=prim, **kwargs)
    else:
        splice_p.bind(addr=addr)
        ret = prim(*args)
        unsplice_p.bind(addr=addr)
        return ret


def trace(addr, prim):
    return lambda *args, **kwargs: _trace(addr, prim, *args, **kwargs)


def trace_abstract_eval(*args, addr, prim, **kwargs):
    prim = prim()
    return prim.abstract_eval(*args, **kwargs)


trace_p.def_abstract_eval(trace_abstract_eval)
trace_p.multiple_results = True
trace_p.must_handle = True

#####
# encapsulated
#####


def encapsulated_abstract_eval(*args, addr, prim, **kwargs):
    return prim.abstract_eval(*args, **kwargs)


encapsulated_p.def_abstract_eval(encapsulated_abstract_eval)
encapsulated_p.multiple_results = True
encapsulated_p.must_handle = True

#####
# splice
#####


def splice_abstract_eval(addr):
    return abstract_arrays.ShapedArray(shape=(0,), dtype=bool)


def splice_fallback(addr):
    return


splice_p.def_impl(splice_fallback)
splice_p.def_abstract_eval(splice_abstract_eval)
splice_p.must_handle = True

#####
# unsplice
#####


def unsplice_abstract_eval(addr):
    return abstract_arrays.ShapedArray(shape=(0,), dtype=bool)


def unsplice_fallback(addr):
    return


unsplice_p.def_impl(splice_fallback)
unsplice_p.def_abstract_eval(unsplice_abstract_eval)
unsplice_p.must_handle = True
