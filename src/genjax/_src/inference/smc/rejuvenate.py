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


from dataclasses import dataclass

import jax

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.inference.mcmc.metropolis_hastings import mh
from genjax._src.inference.smc.state import SMCAlgorithm
from genjax._src.inference.smc.state import SMCState


@dataclass
class SMCProposalMetropolisHastingsRejuvenate(SMCAlgorithm):
    proposal: GenerativeFunction

    def flatten(self):
        return (self.proposal,), ()

    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
        proposal_args: Tuple,
    ) -> SMCState:
        particles = state.get_particles()
        n_particles = state.get_num_particles()
        kernel = mh(self.proposal)
        sub_keys = jax.random.split(key, n_particles)
        _, rejuvenated_particles = jax.vmap(kernel.apply, in_axes=(0, 0, None))(
            sub_keys, particles, proposal_args
        )
        new_state = SMCState(
            n_particles,
            rejuvenated_particles,
            state.log_weights,
            0.0,
            True,
        )
        return new_state


@dispatch
def smc_rejuvenate(proposal: GenerativeFunction):
    return SMCProposalMetropolisHastingsRejuvenate.new(proposal)
