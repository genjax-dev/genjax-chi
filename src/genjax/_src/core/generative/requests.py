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

from genjax._src.core.generative.choice_map import (
    ChoiceMap,
    Selection,
)
from genjax._src.core.generative.core import (
    Argdiffs,
    EditRequest,
    Retdiff,
    Weight,
)
from genjax._src.core.generative.generative_function import Trace
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    IntArray,
    PRNGKey,
    TypeVar,
)

# Type variables
R = TypeVar("R")


@Pytree.dataclass(match_args=True)
class EmptyRequest(EditRequest):
    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        return tr, jnp.array(0.0), Diff.no_change(tr.get_retval()), EmptyRequest()


@Pytree.dataclass(match_args=True)
class Regenerate(EditRequest):
    selection: Selection

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        gen_fn = tr.get_gen_fn()
        return gen_fn.edit(key, tr, self, argdiffs)


@Pytree.dataclass
class Index(EditRequest):
    idx: IntArray
    request: EditRequest

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        gen_fn = tr.get_gen_fn()
        return gen_fn.edit(key, tr, self, argdiffs)


@Pytree.dataclass(match_args=True)
class ChoiceMapRequest(EditRequest):
    request_choice_map: ChoiceMap

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        gen_fn = tr.get_gen_fn()
        return gen_fn.edit(key, tr, self, argdiffs)
