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

from dataclasses import dataclass

import jax.numpy as jnp

from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import LanguageConstructor
from genjax._src.core.interpreters.incremental import tree_diff_no_change
from genjax._src.core.typing import Int
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.vector.map_combinator import (
    map_combinator,
)


@dataclass
class RepeatCombinator(JAXGenerativeFunction):
    repeats: Int
    inner: JAXGenerativeFunction

    def flatten(self):
        return (self.inner,), (self.repeats,)

    def simulate(self, key, args):
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            in_axes=(0, *(None for _ in args)),
        )
        return mapped.simulate(key, (r, *args))

    def importance(self, key, choice, args):
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            in_axes=(0, *(None for _ in args)),
        )
        return mapped.importance(key, choice, (r, *args))

    def update(self, key, prev, choice, argdiffs):
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            in_axes=(0, *(None for _ in argdiffs)),
        )
        return mapped.update(
            key,
            prev,
            choice,
            (tree_diff_no_change(r), *argdiffs),
        )

    def assess(self, choice, args):
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            in_axes=(0, *(None for _ in args)),
        )
        return mapped.assess(choice, (r, *args))


#########################
# Language constructors #
#########################


@typecheck
def repeat_combinator(
    gen_fn: JAXGenerativeFunction,
    num_repeats: Int,
):
    return RepeatCombinator(num_repeats, gen_fn)


Repeat = LanguageConstructor(
    repeat_combinator,
)
