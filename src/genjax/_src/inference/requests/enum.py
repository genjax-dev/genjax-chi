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
import jax.random as jrand
import jax.tree_util as jtu
from jax import vmap
from jax.scipy.special import logsumexp

from genjax._src.core.generative.choice_map import ChoiceMap
from genjax._src.core.generative.core import (
    Argdiffs,
    EditRequest,
    Retdiff,
    Weight,
)
from genjax._src.core.generative.generative_function import (
    GenerativeFunction,
    Trace,
    Update,
)
from genjax._src.core.generative.requests import Wiggle
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    PRNGKey,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    categorical,
)


@Pytree.dataclass(match_args=True)
class Enum(EditRequest):
    """
    The `Enum` edit request is a compositional request which utilizes
    a grid and a "smoothing" proposal to propose a change to a trace.

    Specifying an enumeration requires that a user provide a `smoother` generative function, and an `gridder`, which is a callable that accepts the `ChoiceMap` from the previous trace and produces a vectorized `ChoiceMap`, indicating a set of grid points (as choice maps) to evaluate an `Update` on the provided trace. The weights from these `Update` edits are then used to select a single grid point, which is provided to the `smoother`.

    The job of the `smoother` is to provide a distribution _around the single grid point_ which will be used to generate the final proposal.
    """

    smoother: GenerativeFunction[Any]
    gridder: Callable[[ChoiceMap], ChoiceMap] = Pytree.static()

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[Any],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[Any], Weight, Retdiff[Any], "EditRequest"]:
        chm = tr.get_choices()
        fwd_grid = self.gridder(chm)

        # Should be a unique size.
        (grid_size,) = set(jtu.tree_leaves(jtu.tree_map(lambda v: len(v), fwd_grid)))

        def grid_update(key, tr, chm):
            request = Update(chm)
            new_tr, w, *_ = request.edit(key, tr, argdiffs)
            return new_tr, w

        #####
        # Compute the forward proposal and score (K).
        #####

        key, sub_key = jrand.split(key)
        sub_keys = jrand.split(sub_key, grid_size)
        fwd_grid_traces, ws = vmap(grid_update, in_axes=[0, None, 0])(
            sub_keys,
            tr,
            fwd_grid,
        )
        key, sub_key = jrand.split(key)
        idx = categorical.sample(sub_key, ws)
        fwd_grid_trace = jtu.tree_map(lambda v: v[idx], fwd_grid_traces)
        avg_fwd_weight = logsumexp(ws) - jnp.log(grid_size)

        # Run Wiggle using the smoother in the forward direction.
        request = Wiggle(
            self.smoother,
            lambda _: (fwd_grid_trace.get_choices(),),
        )
        final_tr, fwd_ratio, retdiff, _ = request.edit(key, tr, argdiffs)

        #####
        # Compute the backward proposal and score (L).
        #####

        bwd_grid = self.gridder(chm)
        key, sub_key = jrand.split(key)
        sub_keys = jrand.split(sub_key, grid_size)
        bwd_grid_traces, ws = vmap(grid_update, in_axes=[0, None, 0])(
            sub_keys,
            final_tr,
            bwd_grid,
        )
        key, sub_key = jrand.split(key)
        avg_bwd_weight = logsumexp(ws) - jnp.log(grid_size)
        bwd_grid_trace = jtu.tree_map(lambda v: v[idx], bwd_grid_traces)

        # Run Wiggle using the smoother in the backward direction.
        bwd_request = Wiggle(
            self.smoother,
            lambda _: (bwd_grid_trace.get_choices(),),
        )
        _, bwd_ratio, _, _ = bwd_request.edit(key, tr, argdiffs)
        return (
            final_tr,
            (bwd_ratio - fwd_ratio) + (avg_bwd_weight - avg_fwd_weight),
            retdiff,
            Enum(self.smoother, self.gridder),
        )
