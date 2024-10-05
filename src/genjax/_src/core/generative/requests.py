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
    Retdiff,
    Score,
    Weight,
)
from genjax._src.core.generative.generative_function import (
    EditRequest,
    Tracediff,
    TraceTangent,
    UnitTangent,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    IntArray,
    PRNGKey,
    TypeVar,
)

# Type variables
R = TypeVar("R")


@Pytree.dataclass
class EmptyTracediff(TraceTangent):
    def get_score(self) -> Score:
        return jnp.array(0.0)


@Pytree.dataclass(match_args=True)
class EmptyRequest(EditRequest):
    def edit(
        self,
        key: PRNGKey,
        tracediff: Tracediff[Any, Any],
        argdiffs: Argdiffs,
    ) -> tuple[UnitTangent, Weight, Retdiff[R], "EditRequest"]:
        assert Diff.static_check_no_change(argdiffs)
        trace = tracediff.get_primal()
        return (
            UnitTangent(),
            jnp.array(0.0),
            Diff.no_change(trace.get_retval()),
            EmptyRequest(),
        )


@Pytree.dataclass(match_args=True)
class Regenerate(EditRequest):
    selection: Selection

    def edit(
        self,
        key: PRNGKey,
        tracediff: Tracediff[Any, Any],
        argdiffs: Argdiffs,
    ) -> tuple[TraceTangent, Weight, Retdiff[R], "EditRequest"]:
        trace = tracediff.get_primal()
        gen_fn = trace.get_gen_fn()
        return gen_fn.edit(key, tracediff, self, argdiffs)


@Pytree.dataclass(match_args=True)
class IndexTangent(TraceTangent):
    idx: IntArray
    tangent: TraceTangent

    def __mul__(self, other: TraceTangent) -> TraceTangent:
        match other:
            case IndexTangent(idx, other_subtangent):
                # TODO: we assume that the idx is the same -- it is an error
                # if it is not!
                return IndexTangent(idx, self.tangent * other_subtangent)
            case _:
                raise NotImplementedError

    def get_delta_score(self) -> Score:
        return self.tangent.get_delta_score()


@Pytree.dataclass(match_args=True)
class Index(EditRequest):
    idx: IntArray
    request: EditRequest

    def edit(
        self,
        key: PRNGKey,
        tracediff: Tracediff[Any, Any],
        argdiffs: Argdiffs,
    ) -> tuple[TraceTangent, Weight, Retdiff[R], "EditRequest"]:
        trace = tracediff.get_primal()
        gen_fn = trace.get_gen_fn()
        return gen_fn.edit(key, tracediff, self, argdiffs)


@Pytree.dataclass(match_args=True)
class ChoiceMapRequest(EditRequest):
    request_choice_map: ChoiceMap

    def edit(
        self,
        key: PRNGKey,
        tracediff: Tracediff[Any, Any],
        argdiffs: Argdiffs,
    ) -> tuple[TraceTangent, Weight, Retdiff[R], "EditRequest"]:
        trace = tracediff.get_primal()
        gen_fn = trace.get_gen_fn()
        return gen_fn.edit(key, tracediff, self, argdiffs)
