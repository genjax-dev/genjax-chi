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


from genjax._src.core.generative import (
    GFI,
)
from genjax._src.core.typing import (
    Callable,
    TypeVar,
)

R = TypeVar("R")


def RepeatCombinator(gen_fn: GFI[R], /, *, n: int) -> GFI[R]:
    """
    A combinator that samples from a supplied [`genjax.GFI`][] `gen_fn` a fixed number of times, returning a vector of `n` results.

    See [`genjax.repeat`][] for more details.
    """
    return gen_fn.vmap(in_axes=None, axis_size=n)


def repeat(*, n: int) -> Callable[[GFI[R]], GFI[R]]:
    """
    Returns a decorator that wraps a [`genjax.GFI`][] `gen_fn` of type `a -> b` and returns a new `GFI` of type `a -> [b]` that samples from `gen_fn `n` times, returning a vector of `n` results.

    The values traced by each call `gen_fn` will be nested under an integer index that matches the loop iteration index that generated it.

    This combinator is useful for creating multiple samples from the same generative model in a batched manner.

    Args:
        n: The number of times to sample from the generative function.

    Returns:
        A new [`genjax.GFI`][] that samples from the original function `n` times.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="repeat"
        import genjax, jax


        @genjax.repeat(n=10)
        @genjax.gen
        def normal_draws(mean):
            return genjax.normal(mean, 1.0) @ "x"


        key = jax.random.key(314159)

        # Generate 10 draws from a normal distribution with mean 2.0
        tr = jax.jit(normal_draws.simulate)(key, (2.0,))
        print(tr.render_html())
        ```
    """

    def decorator(gen_fn: GFI[R]) -> GFI[R]:
        return RepeatCombinator(gen_fn, n=n)

    return decorator
