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
import jax.tree_util as jtu
from jax import grad

from genjax._src.core.generative.choice_map import (
    ChoiceMap,
    Selection,
)
from genjax._src.core.generative.core import (
    Argdiffs,
    Retdiff,
    Weight,
)
from genjax._src.core.generative.generative_function import (
    EditRequest,
    Tracediff,
    TraceTangent,
    Update,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    PRNGKey,
    TypeVar,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import normal

# Type variables
R = TypeVar("R")


@Pytree.dataclass
class MALA(EditRequest):
    tau: FloatArray
    selection: Selection

    def random_draw_and_score(self, forward_mu, std):
        num_seeds = len(jtu.tree_leaves(forward_mu))
        seeds = [i + 1 for i in range(num_seeds)]
        seed_tree = jtu.tree_unflatten(jtu.tree_structure(forward_mu), seeds)
        jtu.tree_map(
            lambda mu, seed: normal.sample(seed, mu, std),
            forward_mu,
            seed_tree,
        )
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        tracediff: Tracediff[Any, Any],
        argdiffs: Argdiffs,
    ) -> tuple[TraceTangent, Weight, Retdiff[R], "EditRequest"]:
        assert Diff.static_check_no_change(argdiffs)
        gen_fn = tracediff.get_gen_fn()
        args = tracediff.get_args()
        choices = tracediff.get_choices()
        selected = choices.filter(self.selection)
        dable, nondable = selected.grad_split()

        def grad_fn(dable):
            merged = dable.grad_merge(nondable)
            score, _ = gen_fn.assess(merged, args)
            return score

        nabla_dable = grad(grad_fn)(dable)
        forward_mu = jtu.tree_map(
            lambda v, g: v + self.tau * g,
            dable,
            nabla_dable,
        )
        std = jnp.sqrt(2 * self.tau)
        proposed_dable_values, forward_score = self.random_draw_and_score(
            forward_mu, std
        )
        proposed_values = proposed_dable_values.merge(nondable)

        tangent, w, rd, bwd_request = Update(proposed_values).edit(
            key, tracediff, argdiffs
        )
        discard: ChoiceMap = bwd_request.constraint.choice_map
        raise NotImplementedError


@Pytree.dataclass
class HMC(EditRequest):
    pass
