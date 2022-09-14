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

# This file is just a Python clone of Alex Lew's wonderful
# composable_smc module in GenProx (Julia).

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.special import logsumexp
import numpy as np
from dataclasses import dataclass
from genjax.core.datatypes import Trace, ValueChoiceMap
from genjax.core.pytree import Pytree, tree_unstack
from genjax.core.specialization import concrete_cond
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target
from typing import Union, Any

Int = Union[jnp.int32, np.int32]
Float32 = Union[jnp.float32, np.float32]
Vector = Any
SMCAlgorithm = ProxDistribution


def effective_sample_size(log_normalized_weights: Vector):
    log_ess = -logsumexp(2.0 * log_normalized_weights)
    return jnp.exp(log_ess)


@dataclass
class ParticleCollection(Pytree):
    particles: Trace
    weights: Vector
    lml_est: Float32

    def flatten(self):
        return (self.particles, self.weights, self.lml_est), ()

    def log_marginal_likelihood(self):
        return (
            self.lml_est + logsumexp(self.weights) - np.log(len(self.weights))
        )


@dataclass
class SMCResample(SMCAlgorithm):
    previous: SMCAlgorithm
    ess_threshold: Float32
    how_many: Int

    def num_particles(self):
        return self.how_many

    def final_target(self):
        return self.previous.final_target()

    def run_smc(self, key, retained=None):
        key, collection = self.previous.run_smc(key, retained)
        num_particles = len(collection.weights)
        total_weight = logsumexp(collection.weights)
        log_normalized_weights = collection.weights - total_weight

        def _break(key):
            return key, collection

        def _fallthrough(key):
            normalized_weights = jnp.exp(log_normalized_weights)
            key, sub_key = jax.random.split(key)
            selected_particle_indices = jax.random.categorical(
                sub_key, normalized_weights, shape=(self.how_many,)
            )
            if isinstance(retained, None):
                selected_particle_indices[-1] = num_particles
            particles = jtu.tree_map(
                lambda v: v[selected_particle_indices], collection.particles
            )
            weights = jnp.zeros(self.how_many)
            avg_weight = total_weight - np.log(num_particles)
            return key, ParticleCollection(
                particles, weights, avg_weight + collection.lml_est
            )

        check = (
            effective_sample_size(log_normalized_weights) > self.ess_threshold
        )
        return concrete_cond(check, _break, _fallthrough, key)


@dataclass
class SMCClone(SMCAlgorithm):
    previous: SMCAlgorithm
    factor: Int

    def num_particles(self):
        return self.previous.num_particles() * factor

    def final_target(self):
        return self.previous.final_target()

    def run_smc(self, retained=None):
        pass


@dataclass
class SMCInit(SMCAlgorithm):
    q: Any
    target: Target
    num_particles: Int

    def num_particles(self):
        return self.num_particles

    def final_target(self):
        return self.target

    def run_smc(self, key, retained=None):
        key, sub_keys = jax.random.split(key, self.num_particles + 1)
        _, proposals = jax.vmap(self.q.simulate, in_axes=(0, None))(
            sub_keys,
            (self.target,),
        )
        proposals = tree_unstack(proposals)
        if isinstance(retained, None):
            key, (_, end) = self.q.importance(
                key, ValueChoiceMap(retained), (self.target,)
            )
            proposals[-1] = end
