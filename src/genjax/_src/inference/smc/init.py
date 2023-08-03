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


import jax

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.inference.smc.state import SMCState


@dispatch
def smc_initialize(
    key: PRNGKey,
    model: GenerativeFunction,
    model_args: Tuple,
    obs: ChoiceMap,
    n_particles: Int,
) -> SMCState:
    sub_keys = jax.random.split(key, n_particles)
    (lws, particles) = jax.vmap(model.importance, in_axes=(0, None, None))(
        sub_keys, obs, model_args
    )
    return SMCState(n_particles, particles, lws, 0.0, True)


@dispatch
def smc_initialize(
    key: PRNGKey,
    model: GenerativeFunction,
    model_args: Tuple,
    proposal: GenerativeFunction,
    proposal_args: Tuple,
    obs: ChoiceMap,
    n_particles: Int,
) -> Tuple[PRNGKey, SMCState]:
    key, sub_key = jax.random.split(key)
    sub_keys = jax.random.split(sub_key, n_particles)
    (_, proposal_scores, proposals) = jax.vmap(
        proposal.propose, in_axes=(0, None, None)
    )(sub_keys, obs, proposal_args)

    def _inner(key, proposal):
        constraints = obs.merge(proposal)
        _, (model_score, particle) = model.importance(key, constraints, model_args)
        return model_score, particle

    sub_keys = jax.random.split(key, n_particles)
    model_scores, particles = jax.vmap(_inner)(sub_keys, proposals)
    lws = model_scores - proposal_scores
    return SMCState(n_particles, particles, lws, 0.0, True)
