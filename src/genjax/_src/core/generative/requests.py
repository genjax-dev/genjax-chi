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
    Selection,
)
from genjax._src.core.generative.core import (
    Argdiffs,
    EditRequest,
    PrimitiveEditRequest,
    Retdiff,
    Weight,
)
from genjax._src.core.generative.generative_function import Trace
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    PRNGKey,
    TypeVar,
)

# Type variables
R = TypeVar("R")
ER = TypeVar("ER", bound=EditRequest)


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
class Regenerate(PrimitiveEditRequest):
    selection: Selection


# NOTE: can be used in an unsafe fashion!
@Pytree.dataclass(match_args=True)
class DiffAnnotate(Generic[ER], EditRequest):
    """
    The `DiffAnnotate` request can be used to introspect on the values of type `Diff` (primal and change tangent) values flowing
    through an edit program.

    Users can provide an `argdiff_fn` and a `retdiff_fn` to manipulate changes. Note that, this introspection is inherently unsafe, users should expect:

        * If you convert `Argdiffs` in such a way that you _assert_ that a value hasn't changed (when it actually has), the edit computation will be incorrect. Similar for the `Retdiff`.
    """

    request: ER
    argdiff_fn: Callable[[Argdiffs], Argdiffs] = Pytree.static(default=lambda v: v)
    retdiff_fn: Callable[[Retdiff[Any]], Retdiff[Any]] = Pytree.static(
        default=lambda v: v
    )

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], "EditRequest"]:
        new_argdiffs = self.argdiff_fn(argdiffs)
        tr, w, retdiff, bwd_request = self.request.edit(key, tr, new_argdiffs)
        new_retdiff = self.retdiff_fn(retdiff)
        return tr, w, new_retdiff, bwd_request
