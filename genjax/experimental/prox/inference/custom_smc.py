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
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from genjax.experimental.prox.target import Target
from genjax.experimental.prox.prox_distribution import ProxDistribution
from typing import Any, Callable, Union

Int = Union[np.int32, jnp.int32]


@dataclass
class CustomSMC(ProxDistribution):
    initial_state: Callable[[Target], Any]
    step_model: Callable[[Any, Target], Target]
    step_proposal: ProxDistribution
    num_steps: Callable[[Target], Int]
    num_particles: Int

    def flatten(self):
        return (), (
            self.initial_state,
            self.step_model,
            self.step_proposal,
            self.num_steps,
            self.num_particles,
        )

    def random_weighted(self, key, target):
        init = self.initial_state(target)
        states = jnp.repeat(init, self.num_particles)
        target_weights = jnp.zeros(self.num_particles)
        weights = jnp.zeros(self.num_particles)
        N = self.num_steps(target)

        def _particle_step(key, state):
            new_target = self.step_model(state)
            key, particle = self.step_proposal.simulate(
                key, (state, new_target, target)
            )
            key, (_, new_target_trace) = new_target.importance(
                key, particle.get_retval(), ()
            )
            target_weight = new_target_trace.get_score()
            weight = new_target_trace.get_score() - particle.get_score()
            new_state = new_target_trace.get_retval()
            return key, (new_state, particle, target_weight, weight)

        def _inner(carry, x):
            key, states, target_weights, weights = carry
            key, sub_keys = jax.random.split(key, self.num_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, (particles, target_weights, weights, states) = jax.vmap(
                _particle_step, in_axes=(0, 0)
            )(sub_keys, states)
            total_weight = jax.scipy.special.logsumexp(weights)
            log_normalized_weights = weights - total_weight
            key, sub_key = jax.random.split(key)
            selected_particle_indices = jax.random.categorical(
                sub_key, log_normalized_weights, shape=(self.num_particles)
            )
            target_weights = target_weights[selected_particle_indices]
            average_weight = total_weight - np.log(self.num_particles)
            weights = jnp.repeat(average_weight, self.num_particles)
            states = states[selected_particle_indices]
            return (key, states, target_weights, weights), particles

        (key, states, target_weights, weights), particles = jax.lax.scan(
            _inner,
            (key, states, target_weights, weights),
            None,
            length=N,
        )

        key, sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        final_target_scores = jax.vmap(
            target.importance,
            in_axes=(0, 0, None),
        )(sub_keys, particles, ())
        final_weights = weights - target_weights + final_target_scores
        total_weight = jax.scipy.special.logsumexp(final_weights)
        log_normalized_weights = final_weights - total_weight
        average_weight = total_weight - np.log(self.num_particles)
        key, sub_key = jax.random.split(key)
        selected_particle_index = jax.random.categorical(
            key, log_normalized_weights
        )
        selected_particle = particles[selected_particle_index]
        return key, (
            selected_particle,
            final_target_scores[selected_particle] - average_weight,
        )

    def estimate_logpdf(self, k, v, *args):
        pass
