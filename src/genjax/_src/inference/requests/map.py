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

import jax.tree_util as jtu

from genjax._src.core.generative import (
    Argdiffs,
    EditRequest,
    Retdiff,
    Selection,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.generative.requests import DiffAnnotate
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    PRNGKey,
)
from genjax._src.inference.requests.gradient_utils import (
    selection_gradient,
)

#######
# MAP #
#######


@Pytree.dataclass(match_args=True)
class MAP(EditRequest):
    selection: Selection
    eps: FloatArray

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[Any],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[Any], Weight, Retdiff[Any], "EditRequest"]:
        # A conservative restriction, for now.
        assert Diff.static_check_no_change(argdiffs)
        choice_map_grad = selection_gradient(self.selection, tr, argdiffs)
        original_choice_map = tr.get_choices()
        updated_choices = jtu.tree_map(
            lambda v, g: v + self.eps * g,
            original_choice_map,
            choice_map_grad,
        )
        return Update(updated_choices).edit(key, tr, updated_choices)


def SafeMAP(selection: Selection, eps: FloatArray) -> DiffAnnotate[MAP]:
    def retdiff_assertion(retdiff: Retdiff[Any]):
        assert Diff.static_check_no_change(retdiff)
        return retdiff

    return MAP(selection, eps).map(retdiff_assertion)
