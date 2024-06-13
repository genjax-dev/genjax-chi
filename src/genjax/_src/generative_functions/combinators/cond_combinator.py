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
from genjax._src.core.typing import ScalarBool, typecheck
from genjax._src.generative_functions.combinators.compose_combinator import (
    ComposeCombinator,
)
from genjax._src.generative_functions.combinators.switch_combinator import (
    SwitchCombinator,
)

register_exclusion(__file__)


@typecheck
def CondCombinator(
    if_gen_fn: GenerativeFunction,
    else_gen_fn: GenerativeFunction,
) -> ComposeCombinator:
    @typecheck
    def argument_mapping(b: ScalarBool, *args):
        # Note that `True` maps to 0 to select the "if" branch, `False` to 1.
        idx = jnp.array(jnp.logical_not(b), dtype=int)
        return (idx, *args)

    inner_combinator = SwitchCombinator((if_gen_fn, else_gen_fn))

    return ComposeCombinator(
        inner_combinator,
        argument_mapping=argument_mapping,
        retval_mapping=lambda _, retval: retval,
        info="Derived combinator (Cond)",
    )


@typecheck
def cond_combinator(
    if_gen_fn: GenerativeFunction,
    else_gen_fn: GenerativeFunction,
) -> ComposeCombinator:
    return CondCombinator(if_gen_fn, else_gen_fn)
