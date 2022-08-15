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

# This example showcases the `incremental` interface.

import jax
import jax.numpy as jnp
import genjax as gex

# A function which sends off `sum` to `jnp`
def sum(x):
    return 3 * jnp.sum(x)


# Here, `ArrayAppend` is a `Cell` -- a type which
# is registered with the incremental interpreter and allows
# abstract interpretation.
df = gex.incremental(sum, gex.ArrayAppend)

# Here, `df` is `d_sum x dx` -- it takes a value
# `x` in the base value space, and `dx` a value in the
# `gex.ArrayAppend` space (the change type).
#
# The result is the incremental derivative of `f`.
#
# For a function g, to determine g(x ⊕ dx), the user must evaluate
# g(x) ⊕  dg x dx.
#
# The incremental derivative sometimes needs the base value for certain
# computations (non-maintainable computations).
#
# But in general, computing the updated value will be as fast as full
# re-evaluation.

res = jax.jit(sum)(5) + jax.jit(df)(5, 2)
