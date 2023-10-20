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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def get_trace_data_shape(gen_fn, key, args):
    def _apply(key, args):
        tr = gen_fn.simulate(key, args)
        return tr

    (_, trace_shape) = jax.make_jaxpr(_apply, return_shape=True)(key, args)
    return trace_shape


def get_discard_data_shape(gen_fn, key, tr, constraints, argdiffs):
    def _apply(key, tr, constraints, argdiffs):
        _, _, _, discard = gen_fn.update(key, tr, constraints, argdiffs)
        return discard

    (_, discard_shape) = jax.make_jaxpr(_apply, return_shape=True)(
        key, tr, constraints, argdiffs
    )
    return discard_shape


def make_zero_trace(gen_fn, *args):
    out_tree = get_trace_data_shape(gen_fn, *args)
    return jtu.tree_map(
        lambda v: jnp.zeros(v.shape, v.dtype),
        out_tree,
    )
