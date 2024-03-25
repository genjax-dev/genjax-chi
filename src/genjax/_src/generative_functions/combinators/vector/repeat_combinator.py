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

from typing import Callable

import jax
import jax.numpy as jnp

from genjax._src.core.datatypes.generative import (
    Choice,
    GenerativeFunction,
    HierarchicalSelection,
    JAXGenerativeFunction,
    Trace,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    Int,
    PRNGKey,
    Tuple,
    dispatch,
    typecheck,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexedSelection,
    VectorChoiceMap,
)
from genjax._src.generative_functions.static.static_gen_fn import SupportsCalleeSugar


class RepeatTrace(Trace):
    repeated_fn: GenerativeFunction
    inner_trace: Trace
    args: Tuple

    def get_args(self):
        return self.args

    def get_choices(self):
        return VectorChoiceMap(self.inner_trace.strip())

    def get_gen_fn(self):
        return self.repeated_fn

    def get_score(self):
        return jnp.sum(jax.vmap(lambda it: it.get_score())(self.inner_trace))

    def get_retval(self):
        return self.inner_trace.get_retval()

    @dispatch
    def project(self, selection: IndexedSelection):
        raise NotImplementedError()

    @dispatch
    def project(self, selection: HierarchicalSelection):
        return jnp.sum(jax.vmap(lambda it: it.project(selection))(self.inner_trace))


class RepeatCombinator(
    SupportsCalleeSugar,
    JAXGenerativeFunction,
):
    """The `RepeatCombinator` supports i.i.d sampling from generative functions (for
    vectorized mapping over arguments, see `MapCombinator`)."""

    repeats: Int = Pytree.static()
    inner: JAXGenerativeFunction

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> RepeatTrace:
        sub_keys = jax.random.split(key, self.repeats)
        repeated_inner_tr = jax.vmap(
            self.inner.simulate,
            in_axes=(0, None),
        )(sub_keys, args)
        return RepeatTrace(self.inner, repeated_inner_tr, args)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        choice: VectorChoiceMap,
        args: Tuple,
    ) -> Tuple[RepeatTrace, FloatArray]:
        sub_keys = jax.random.split(key, self.repeats)
        repeated_inner_tr, w = jax.vmap(
            self.inner.importance,
            in_axes=(0, 0, None),
        )(sub_keys, choice.inner, args)
        return RepeatTrace(self.inner, repeated_inner_tr, args), jnp.sum(w)

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: RepeatTrace,
        choice: Choice,
        argdiffs: Tuple,
    ) -> Tuple[RepeatTrace, FloatArray, Any, Choice]:
        pass

    @dispatch
    def assess(
        self,
        choice: VectorChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        inner_choice = choice.inner
        (ws, r) = jax.vmap(self.inner.assess, in_axes=(0, None))(inner_choice, args)
        return jnp.sum(ws), r


#############
# Decorator #
#############


def repeat_combinator(*, repeats) -> Callable[[Callable], JAXGenerativeFunction]:
    def decorator(f) -> JAXGenerativeFunction:
        return RepeatCombinator(repeats, f)

    return decorator
