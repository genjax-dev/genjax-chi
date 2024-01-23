# Copyright 2023 MIT Probabilistic Computing Project
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

from genjax._src.core.datatypes.generative import ChoiceMap, GenerativeFunction
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import (
    Int,
    Optional,
    PRNGKey,
    Tuple,
    dispatch,
    static_check_is_concrete,
)
from genjax._src.inference.smc.state import SMCAlgorithm, SMCState


class SMCInitializeFromPrior(SMCAlgorithm):
    n_particles: Int = Pytree.static()
    model: GenerativeFunction

    def apply(
        self,
        key: PRNGKey,
        obs: ChoiceMap,
        model_args: Tuple,
    ):
        sub_keys = jax.random.split(key, self.n_particles)
        (particles, lws) = jax.vmap(self.model.importance, in_axes=(0, None, None))(
            sub_keys, obs, model_args
        )
        return SMCState(self.n_particles, particles, lws, 0.0, True)


class SMCInitializeFromProposal(SMCAlgorithm):
    n_particles: Int = Pytree.static()
    model: GenerativeFunction
    proposal: GenerativeFunction

    def __post_init__(self):
        assert static_check_is_concrete(self.n_particles)

    def apply(
        self,
        key: PRNGKey,
        obs: ChoiceMap,
        model_args: Tuple,
        proposal_args: Tuple,
    ):
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.n_particles)
        (_, proposal_scores, proposals) = jax.vmap(
            self.proposal.propose, in_axes=(0, None, None)
        )(sub_keys, obs, proposal_args)

        def _inner(key, proposal):
            constraints = obs.merge(proposal)
            (particle, model_score) = self.model.importance(
                key, constraints, model_args
            )
            return model_score, particle

        sub_keys = jax.random.split(key, self.n_particles)
        model_scores, particles = jax.vmap(_inner)(sub_keys, proposals)
        lws = model_scores - proposal_scores
        return SMCState(self.n_particles, particles, lws, 0.0, True)


@dispatch
def smc_initialize(
    model: GenerativeFunction,
    proposal: Optional[GenerativeFunction] = None,
    *,
    n_particles: int,
) -> SMCAlgorithm:
    if proposal is not None:
        return SMCInitializeFromProposal(n_particles, model, proposal)
    else:
        return SMCInitializeFromPrior(n_particles, model)
