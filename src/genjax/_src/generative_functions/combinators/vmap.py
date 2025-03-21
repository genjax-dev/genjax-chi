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
"""The `Vmap` is a generative function combinator which exposes vectorization
on the input arguments of a provided generative function callee.

This vectorization is implemented using `jax.vmap`, and the combinator expects the user to specify `in_axes` as part of the construction of an instance of this combinator.
"""

import jax.numpy as jnp

from genjax._src.core.generative import (
    GFI,
    Argdiffs,
    ChoiceMap,
    EditRequest,
    R,
    Retdiff,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.generative.choice_map import (
    Selection,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    InAxes,
)
from genjax.pjax import vmap as modular_vmap


@Pytree.dataclass
class Vmap(Generic[R], GFI[R]):
    """`Vmap` is a generative function which lifts another generative function to support `vmap`-based patterns of parallel (and generative) computation.

    In contrast to the full set of options which [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), this combinator expects an `in_axes: tuple` configuration argument, which indicates how the underlying `vmap` patterns should be broadcast across the input arguments to the generative function.

    Attributes:
        gen_fn: A [`genjax.GFI`][] to be vectorized.

        in_axes: A tuple specifying which input arguments (or indices into them) should be vectorized. `in_axes` must match (or prefix) the `Pytree` type of the argument tuple for the underlying `gen_fn`. Defaults to 0, i.e., the first argument. See [this link](https://jax/readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees) for more detail.

    Examples:
        Create a `Vmap` using the [`genjax.vmap`][] decorator:
        ```python exec="yes" html="true" source="material-block" session="vmap"
        import jax, genjax
        import jax.numpy as jnp


        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def mapped(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        key = jax.random.key(314159)
        arr = jnp.ones(100)

        tr = jax.jit(mapped.simulate)(key, (arr,))
        print(tr.render_html())
        ```

        Use the [`genjax.GFI.vmap`][] method:
        ```python exec="yes" html="true" source="material-block" session="vmap"
        @genjax.gen
        def add_normal_noise(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        mapped = add_normal_noise.vmap(in_axes=(0,))

        tr = jax.jit(mapped.simulate)(key, (arr,))
        print(tr.render_html())
        ```
    """

    gen_fn: GFI[R]
    in_axes: InAxes = Pytree.static()
    axis_size: int | None = Pytree.static()
    axis_name: str | None = Pytree.static()
    spmd_axis_name: str | None = Pytree.static()

    def __abstract_call__(self, *args) -> Any:
        return modular_vmap(
            self.gen_fn.__abstract_call__,
            in_axes=self.in_axes,
        )(*args)

    def simulate(
        self,
        args: tuple[Any, ...],
    ) -> Trace[R]:
        # vmapping over `gen_fn`'s `simulate` gives us a new trace with vector-shaped leaves.
        tr = modular_vmap(
            self.gen_fn.simulate,
            in_axes=self.in_axes,
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(args)
        return tr

    def assess(
        self,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        scores, retvals = modular_vmap(
            self.gen_fn.assess,
            in_axes=(0, self.in_axes),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(chm, args)
        return jnp.sum(scores), retvals

    def generate(
        self,
        constraint: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Trace[R], Weight]:
        tr, weight_v = modular_vmap(
            self.gen_fn.generate,
            in_axes=(0, self.in_axes),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.spmd_axis_name,
        )(constraint, args)
        w = jnp.sum(weight_v)
        return tr, w

    def project(
        self,
        trace: Trace[R],
        selection: Selection,
    ) -> Weight:
        raise NotImplementedError

    def edit(
        self,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        raise NotImplementedError


#############
# Decorator #
#############


def vmap(
    *,
    in_axes: InAxes = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
) -> Callable[[GFI[R]], Vmap[R]]:
    """
    Returns a decorator that wraps a [`GFI`][genjax.GFI] and returns a new `GFI` that performs a vectorized map over the argument specified by `in_axes`. Traced values are nested under an index, and the retval is vectorized.

    Args:
        in_axes: Selector specifying which input arguments (or index into them) should be vectorized. `in_axes` must match (or prefix) the `Pytree` type of the argument tuple for the underlying `gen_fn`. Defaults to 0, i.e., the first argument. See [this link](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees) for more detail.

    Returns:
        A decorator that converts a [`genjax.GFI`][] into a new [`genjax.GFI`][] that accepts an argument of one-higher dimension at the position specified by `in_axes`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="vmap"
        import jax, genjax
        import jax.numpy as jnp


        @genjax.vmap(in_axes=0)
        @genjax.gen
        def vmapped_model(x):
            v = genjax.normal(x, 1.0) @ "v"
            return genjax.normal(v, 0.01) @ "q"


        key = jax.random.key(314159)
        arr = jnp.ones(100)

        # `vmapped_model` accepts an array of numbers:
        tr = jax.jit(vmapped_model.simulate)(key, (arr,))

        print(tr.render_html())
        ```
    """

    def decorator(gen_fn: GFI[R]) -> Vmap[R]:
        return Vmap(
            gen_fn,
            in_axes,
            axis_size,
            axis_name,
            spmd_axis_name,
        )

    return decorator
