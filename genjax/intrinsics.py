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
from jax.interpreters import batching
import inspect
from genjax.primitive import PrimitiveGenerativeFunction

#####
# Primitives
#####

# GenJAX language trace primitive.
trace_p = core.Primitive("trace")

# GenJAX language batch trace primitive.
batched_trace_p = core.Primitive("batched_trace")

# External generative function trace primitive.
primitive_p = core.Primitive("primitive")

# Hierarchical (call) addressing primitive.
splice_p = core.Primitive("splice")

# Hierarchical (call) addressing primitive.
unsplice_p = core.Primitive("unsplice")

#####
# trace
#####


def _trace(addr, call, *args, **kwargs):
    if isinstance(call, PrimitiveGenerativeFunction):
        return primitive_p.bind(*args, addr=addr, prim=call, **kwargs)
    else:
        splice_p.bind(addr=addr)
        key, ret = call(*args)
        unsplice_p.bind(addr=addr)
        return key, ret


def trace(addr, call):
    return lambda *args, **kwargs: _trace(addr, call, *args, **kwargs)


# This defers abstract evaluation to `batched_trace`
# where it can pass new batched shapes to primitives
# via the `shape` keyword argument.
def trace_batch(args, batch_axes, addr, call, **kwargs):
    (key, res) = batched_trace_p.bind(
        *args, batch_axes=batch_axes, addr=addr, call=call, **kwargs
    )
    # TODO: check that this batch_axes stuff is correct.
    return (key, res), (batch_axes[1], *batch_axes[2:])


trace_p.multiple_results = True
trace_p.must_handle = True
batching.primitive_batchers[trace_p] = trace_batch

#####
# batched_trace
#####

# This primitive is used to automatically perform the correct
# abstract evaluation for generative functions inside of `vmap` and `pmap`


def batched_trace_abstract_eval(*args, addr, prim, **kwargs):
    prim = prim()
    # This is the number of keys provided.
    batch_dim = args[0].shape[0]
    return prim.abstract_eval_batched(*args, batch_dim=batch_dim, **kwargs)


batched_trace_p.def_abstract_eval(batched_trace_abstract_eval)
batched_trace_p.multiple_results = True
batched_trace_p.must_handle = True


#####
# primitive
#####


def primitive_abstract_eval(*args, addr, prim, **kwargs):
    return prim.abstract_eval(*args, **kwargs)


primitive_p.def_abstract_eval(primitive_abstract_eval)
primitive_p.multiple_results = True
primitive_p.must_handle = True

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
