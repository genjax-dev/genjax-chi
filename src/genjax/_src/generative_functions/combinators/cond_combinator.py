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

from genjax._src.core.generative import GenerativeFunctionClosure
from genjax._src.generative_functions.combinators.compose_combinator import (
    compose_combinator,
)
from genjax._src.generative_functions.combinators.switch_combinator import (
    switch_combinator,
)


def cond_combinator(
    if_gen_fn: GenerativeFunctionClosure,
    else_gen_fn: GenerativeFunctionClosure,
) -> GenerativeFunctionClosure:
    def argument_pushforward(b, *args):
        idx = jnp.array(b, dtype=int)
        return (idx, *args)

    inner_combinator_closure = switch_combinator(if_gen_fn, else_gen_fn)

    def retval_pushforward(args, sample, retval):
        return retval

    return compose_combinator(
        inner_combinator_closure,
        argument_pushforward,
        retval_pushforward,
        info="CondCombinator",
    )
