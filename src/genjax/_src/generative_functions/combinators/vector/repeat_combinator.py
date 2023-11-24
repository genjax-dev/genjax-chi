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

from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import LanguageConstructor
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.interpreters.incremental import tree_diff_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.vector.map_combinator import MapTrace
from genjax._src.generative_functions.combinators.vector.map_combinator import (
    map_combinator,
)


@dataclass
class RepeatTrace(Trace):
    map_trace: MapTrace

    def flatten(self):
        return (self.map_trace,), ()

    def get_score(self):
        return self.map_trace.get_score()

    def get_args(self):
        return self.args

    def get_choices(self):
        return self.map_trace.strip()

    def get_retval(self):
        return self.map_trace.get_retval()

    @typecheck
    def project(self, selection: Selection):
        return self.map_trace.project(selection)


@dataclass
class RepeatCombinator(JAXGenerativeFunction):
    repeats: Int
    inner: JAXGenerativeFunction

    def flatten(self):
        return (self.inner,), (self.repeats,)

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> RepeatTrace:
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            (0, *(None for _ in args)),
        )
        map_trace = mapped.simulate(key, (r, *args))
        return RepeatTrace(map_trace, args)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        choice: Choice,
        args: Tuple,
    ) -> Tuple[RepeatTrace, FloatArray]:
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            (0, *(None for _ in args)),
        )
        (map_trace, w) = mapped.importance(key, choice, (r, *args))
        return RepeatTrace(map_trace, args), w

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: RepeatTrace,
        choice: Choice,
        argdiffs: Tuple,
    ) -> Tuple[RepeatTrace, FloatArray, Any, Choice]:
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            (0, *(None for _ in argdiffs)),
        )
        (map_trace, w, rd, d) = mapped.update(
            key,
            prev,
            choice,
            (tree_diff_no_change(r), *argdiffs),
        )
        args = tree_diff_primal(argdiffs)
        return RepeatTrace(map_trace, args), w, rd, d

    @typecheck
    def assess(
        self,
        choice: Choice,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        r = jnp.arange(self.repeats)
        mapped = map_combinator(
            self.inner,
            (0, *(None for _ in args)),
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
