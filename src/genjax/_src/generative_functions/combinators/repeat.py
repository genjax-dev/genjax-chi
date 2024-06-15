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
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import Callable, Int, IntArray, Tuple, typecheck
from genjax._src.generative_functions.combinators.address_bijection import (
    map_addresses,
)
from genjax._src.generative_functions.combinators.compose import (
    ComposeCombinator,
)
from genjax._src.generative_functions.combinators.vmap import (
    VmapCombinator,
)
from genjax._src.generative_functions.static import gen

register_exclusion(__file__)


def RepeatCombinator(gen_fn: GenerativeFunction, /, *, n: Int) -> ComposeCombinator:
    def argument_mapping(*args):
        return (jnp.zeros(n), args)

    # This is a static generative function which an attached
    # choice map address bijection, to collapse the `_internal`
    # address hierarchy below.
    # (as part of StaticGenerativeFunction.Trace interfaces)
    @map_addresses(mapping={...: "_internal"})
    @gen
    def expanded_gen_fn(_: IntArray, args: Tuple):
        return gen_fn(*args) @ "_internal"

    inner_combinator_closure = VmapCombinator(expanded_gen_fn, in_axes=(0, None))

    return ComposeCombinator(
        inner_combinator_closure,
        argument_mapping,
        lambda _, retval: retval,
        "Derived combinator (Repeat)",
    )


@typecheck
def repeat(n: Int) -> Callable[[GenerativeFunction], ComposeCombinator]:
    def decorator(gen_fn) -> ComposeCombinator:
        return RepeatCombinator(gen_fn, n=n)

    return decorator
