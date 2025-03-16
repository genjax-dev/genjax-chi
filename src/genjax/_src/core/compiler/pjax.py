# Copyright 2024 MIT Probabilistic Computing Project
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

import jax.random as jrand
import jax.tree_util as jtu
from jax import vmap

from genjax._src.core.compiler.initial_style_primitive import (
    InitialStylePrimitive,
    NotEliminatedException,
    initial_style_bind,
)
from genjax._src.core.typing import Any, Callable, PRNGKey

######################
# Sampling primitive #
######################

sample_p = InitialStylePrimitive("sample")


def static_dim_length(in_axes, args: tuple[Any, ...]) -> int:
    # perform the in_axes massaging that vmap performs internally:
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)
    elif isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    def find_axis_size(axis: int | None, x: Any) -> int | None:
        """Find the size of the axis specified by `axis` for the argument `x`."""
        if axis is not None:
            leaf = jtu.tree_leaves(x)[0]
            return leaf.shape[axis]

    # tree_map uses in_axes as a template. To have passed vmap validation, Any non-None entry
    # must bottom out in an array-shaped leaf, and all such leafs must have the same size for
    # the specified dimension. Fetching the first is sufficient.
    axis_sizes = jtu.tree_map(
        find_axis_size,
        in_axes,
        args,
        is_leaf=lambda x: x is None,
    )
    return jtu.tree_leaves(axis_sizes)[0]


# This is very cheeky.
@dataclass
class GlobalKeyCounter:
    count: int = 0


# Very large source of unique keys.
global_counter = GlobalKeyCounter()


def sample_binder(
    jax_impl: Callable[[PRNGKey, Any], Any],
    **kwargs,
):
    def sampler(*args):
        def keyless_jax_impl(*args):
            global_counter.count += 1
            return jax_impl(jrand.PRNGKey(global_counter.count), *args)

        def raise_exception():
            raise NotEliminatedException(
                "JAX is attempting to invoke the implementation of a sampler defined using the `sample_p` primitive in your function.\n\nEliminate `sample_p` in `your_fn` by using the `genjax.pjax(your_fn, key: PRNGKey)(*your_args)` transformation, which allows you to use the JAX implementation of the sampler."
            )

        # Holy smokes recursion.
        def batch(vector_args, batch_axes):
            def batch_jax_impl(key, *args):
                n = static_dim_length(batch_axes, args)
                ks = jrand.split(key, n)
                return vmap(jax_impl, in_axes=(0, *batch_axes))(ks, *args)

            batched_sampler = sample_binder(batch_jax_impl)
            v = batched_sampler(*vector_args)
            return (v,), (0,)

        return initial_style_bind(
            sample_p,
            jax_impl=jax_impl,
            batch=batch,
            raise_exception=raise_exception,
            **kwargs,
        )(keyless_jax_impl)(*args)

    return sampler
