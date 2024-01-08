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

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.typing import PRNGKey, Tuple, typecheck
from genjax._src.inference.smc.state import SMCAlgorithm, SMCState
from genjax._src.inference.smc.utils import dynamic_check_empty
from genjax._src.inference.translator import ExtendingTraceTranslator

############################
# Step forward using prior #
############################


@dataclass
class SMCForwardUpdate(SMCAlgorithm):
    def flatten(self):
        return (), ()

    @typecheck
    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
        new_argdiffs: Tuple,
        obs: ChoiceMap,
    ) -> SMCState:
        target_model = state.get_target_gen_fn()
        particles = state.get_particles()
        n_particles = state.get_num_particles()
        sub_keys = jax.random.split(key, n_particles)
        (particles, log_weights, _, discard) = jax.vmap(
            target_model.update, in_axes=(0, 0, None, None)
        )(sub_keys, particles, obs, new_argdiffs)
        dynamic_check_empty(discard)
        new_state = SMCState(
            n_particles,
            particles,
            state.log_weights + log_weights,
            0.0,
            dynamic_check_empty(discard),
        )
        return new_state


#####################################
# Step forward using extension step #
#####################################


@dataclass
class SMCExtendUpdate(SMCAlgorithm):
    translator: ExtendingTraceTranslator

    def flatten(self):
        return (self.translator), ()

    @typecheck
    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
    ) -> SMCState:
        particles = state.get_particles()
        n_particles = state.get_num_particles()
        sub_keys = jax.random.split(key, n_particles)
        (particles, log_weights) = jax.vmap(self.translator.apply)(sub_keys, particles)
        new_state = SMCState(
            n_particles,
            particles,
            state.log_weights + log_weights,
            0.0,
            state.valid,  # TODO: extend translators with dynamic checks.
        )
        return new_state
