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

from genjax._src.core.generative import (
    GenerativeFunction,
)
from genjax._src.core.typing import Callable, Int, IntArray, Optional, Tuple, typecheck
from genjax._src.generative_functions.combinators.address_bijection_combinator import (
    address_bijection_combinator,
)
from genjax._src.generative_functions.combinators.compose_combinator import (
    compose_combinator,
)
from genjax._src.generative_functions.combinators.vmap_combinator import (
    vmap_combinator,
)
from genjax._src.generative_functions.static.static_gen_fn import static_gen_fn


@typecheck
def repeat_combinator(
    gen_fn: Optional[GenerativeFunction] = None,
    /,
    *,
    num_repeats: Int,
) -> Callable | GenerativeFunction:
    def decorator(gen_fn):
        def argument_pushforward(*args):
            return (jnp.zeros(num_repeats), args)

        # This is a static generative function which an attached
        # choice map address bijection, to collapse the `_internal`
        # address hierarchy below.
        # (as part of StaticGenerativeFunction.Trace interfaces)
        @address_bijection_combinator(address_bijection={(): "_internal"})
        @static_gen_fn
        def expanded_gen_fn(idx: IntArray, args: Tuple):
            return gen_fn(*args) @ "_internal"

        inner_combinator_closure = vmap_combinator(in_axes=(0, None))(expanded_gen_fn)

        return compose_combinator(
            inner_combinator_closure,
            pre=argument_pushforward,
            info="Derived combinator (Repeat)",
        )

    if gen_fn:
        return decorator(gen_fn)
    else:
        return decorator
