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

import jax.numpy as jnp

from genjax._src.core.generative import GenerativeFunction
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import Callable, ScalarBool, Tuple, typecheck

register_exclusion(__file__)


@typecheck
def OrElseCombinator(
    if_gen_fn: GenerativeFunction,
    else_gen_fn: GenerativeFunction,
) -> GenerativeFunction:
    """
    Combinator which enables conditional execution of generative functions. See [`genjax.or_else`][] for more details.

    (This version accepts by `if_gen_fn` and `else_gen_fn` and returns a combinator, vs [`genjax.or_else`][] which returns a decorator.)
    """

    @typecheck
    def argument_mapping(b: ScalarBool, if_args: Tuple, else_args: Tuple):
        # Note that `True` maps to 0 to select the "if" branch, `False` to 1.
        idx = jnp.array(jnp.logical_not(b), dtype=int)
        return (idx, if_args, else_args)

    return if_gen_fn.switch(else_gen_fn).contramap(
        argument_mapping, info="Derived combinator (OrElse)"
    )


@typecheck
def or_else(
    else_gen_fn: GenerativeFunction,
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """
    Returns a decorator that wraps a [`GenerativeFunction`][genjax.GenerativeFunction] `if_gen_fn` and returns a new `GenerativeFunction` that accepts

    - a boolean argument
    - an argument tuple for `if_gen_fn`
    - an argument tuple for the supplied `else_gen_fn`

    and acts like `if_gen_fn` when the boolean is `True` or `else_gen_fn` otherwise.

    Args:
        else_gen_fn: called when the boolean argument is `False`.

    Returns:
        A decorator that produces a new [`GenerativeFunction`][genjax.GenerativeFunction].

    Examples:
        ```python exec="yes" html="true" source="material-block" session="gen-fn"
        import jax
        import jax.numpy as jnp
        import genjax


        @genjax.gen
        def else_model(x):
            return genjax.normal(x, 5.0) @ "else_value"


        @genjax.or_else(else_model)
        @genjax.gen
        def or_else_model(x):
            return genjax.normal(x, 1.0) @ "if_value"


        @genjax.gen
        def model(toss: bool):
            # Note that `or_else_model` takes a new boolean predicate in
            # addition to argument tuples for each branch.
            return or_else_model(toss, (1.0,), (10.0,)) @ "tossed"


        key = jax.random.PRNGKey(314159)

        tr = jax.jit(model.simulate)(key, (True,))

        print(tr.render_html())
        ```
    """

    def decorator(if_gen_fn) -> GenerativeFunction:
        return OrElseCombinator(if_gen_fn, else_gen_fn)

    return decorator
