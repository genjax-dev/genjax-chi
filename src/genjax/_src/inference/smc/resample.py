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
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import dispatch
from genjax._src.inference.smc.state import SMCAlgorithm
from genjax._src.inference.smc.state import SMCState


######################
# Resampling methods #
######################


@dataclass
class ResamplingMethod(Pytree):
    def flatten(self):
        return (), ()


@dataclass
class MultinomialResampling(ResamplingMethod):
    pass


multinomial_resampling = MultinomialResampling()

#####################
# SMC resample step #
#####################


@dataclass
class SMCResample(SMCAlgorithm):
    resampling_method: ResamplingMethod

    def flatten(self):
        return (self.resampling_method,), ()

    @classmethod
    def new(cls, resampling_method):
        return SMCResample(resampling_method)

    @dispatch
    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
        _: MultinomialResampling,
    ) -> SMCState:
        lml_est = state.current_lml_est()
        lws = state.get_log_weights()
        particles = state.get_particles()
        n_particles = state.get_num_particles()
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        idxs = jax.random.categorical(key, log_normalized_weights, shape=(n_particles,))
        resampled_particles = jtu.tree_map(lambda v: v[idxs], particles)
        new_state = SMCState(
            n_particles, resampled_particles, jnp.zeros_like(lws), lml_est, True
        )
        return new_state

    @dispatch
    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
    ) -> SMCState:
        return self.apply(key, state, self.resampling_method)


@dispatch
def smc_resample(resampling_method: ResamplingMethod):
    return SMCResample.new(resampling_method)
