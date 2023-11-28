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

import jax

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import static_check_is_concrete
from genjax._src.inference.smc.state import SMCAlgorithm
from genjax._src.inference.smc.state import SMCState


@dataclass
class SMCInitializeFromPrior(SMCAlgorithm):
    n_particles: Int
    model: GenerativeFunction

    def flatten(self):
        return (self.model,), (self.n_particles,)

    @classmethod
    def new(
        cls,
        model: GenerativeFunction,
        n_particles: Int,
    ):
        return SMCInitializeFromPrior(n_particles, model)

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


@dispatch
def smc_initialize(
    model: GenerativeFunction,
    n_particles: Int,
) -> SMCAlgorithm:
    return SMCInitializeFromPrior.new(model, n_particles)


@dataclass
class SMCInitializeFromProposal(SMCAlgorithm):
    n_particles: Int
    model: GenerativeFunction
    proposal: GenerativeFunction

    def flatten(self):
        return (self.model, self.proposal), (self.n_particles,)

    @classmethod
    def new(
        cls,
        model: GenerativeFunction,
        proposal: GenerativeFunction,
        n_particles: Int,
    ):
        assert static_check_is_concrete(n_particles)
        return SMCInitializeFromPrior(n_particles, model, proposal)

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
    proposal: GenerativeFunction,
    n_particles: Int,
) -> SMCAlgorithm:
    return SMCInitializeFromProposal.new(model, proposal, n_particles)
