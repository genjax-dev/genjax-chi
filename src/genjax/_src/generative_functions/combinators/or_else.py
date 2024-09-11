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
from genjax._src.core.typing import Any, ScalarFlag, TypeVar

R = TypeVar("R")


def or_else(
    if_gen_fn: GenerativeFunction[R],
    else_gen_fn: GenerativeFunction[R],
) -> GenerativeFunction[R]:
    """
    Given two [`genjax.GenerativeFunction`][]s `if_gen_fn` and `else_gen_fn`, returns a new [`genjax.GenerativeFunction`][] that accepts

    - a boolean argument
    - an argument tuple for `if_gen_fn`
    - an argument tuple for the supplied `else_gen_fn`

    and acts like `if_gen_fn` when the boolean is `True` or `else_gen_fn` otherwise.

    Args:
        else_gen_fn: called when the boolean argument is `False`.

    Returns:
        A [`genjax.GenerativeFunction`][] modified for conditional execution.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="or_else"
        import jax
        import jax.numpy as jnp
        import genjax


        @genjax.gen
        def if_model(x):
            return genjax.normal(x, 1.0) @ "if_value"


        @genjax.gen
        def else_model(x):
            return genjax.normal(x, 5.0) @ "else_value"


        or_else_model = genjax.or_else(if_model, else_model)


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

    def argument_mapping(
        b: ScalarFlag, if_args: tuple[Any, ...], else_args: tuple[Any, ...]
    ):
        # Note that `True` maps to 0 to select the "if" branch, `False` to 1.
        idx = jnp.array(jnp.logical_not(b), dtype=int)
        return (idx, if_args, else_args)

    return if_gen_fn.switch(else_gen_fn).contramap(
        argument_mapping, info="Derived combinator (OrElse)"
    )
