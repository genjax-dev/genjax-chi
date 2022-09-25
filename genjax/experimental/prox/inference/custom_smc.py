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

"""
This module supports a JAX compatible implementation of SMC as a :code:`ProxDistribution`.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from dataclasses import dataclass
from genjax.experimental.prox.target import Target
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.combinators.combinator_datatypes import VectorChoiceMap
from typing import Any, Callable, Union

Int = Union[np.int32, jnp.int32]


@dataclass
class CustomSMC(ProxDistribution):
    initial_state: Callable[[Target], Any]
    step_target: Callable[[Any, Any, Target], Target]
    step_proposal: ProxDistribution
    num_steps: Callable[[Target], Int]
    num_particles: Int

    def flatten(self):
        return (), (
            self.initial_state,
            self.step_target,
            self.step_proposal,
            self.num_steps,
            self.num_particles,
        )

    def random_weighted(self, key, final_target):
        init = self.initial_state(final_target)
        states = jnp.repeat(init, self.num_particles)
        target_weights = jnp.zeros(self.num_particles)
        weights = jnp.zeros(self.num_particles)
        constraints = final_target.constraints
        N = self.num_steps(final_target)

        def _particle_step(key, constraints, state):
            new_target = self.step_target(state, constraints, final_target)
            key, particle = self.step_proposal.simulate(
                key, (state, new_target, final_target)
            )
            (particle_chm,) = particle.get_retval()
            key, (_, new_target_trace) = new_target.importance(
                key, particle_chm, ()
            )
            target_weight = new_target_trace.get_score()
            weight = new_target_trace.get_score() - particle.get_score()
            (new_state,) = new_target_trace.get_retval()
            return key, (new_state, particle, target_weight, weight)

        def _inner(carry, x):
            key, states, target_weights, weights = carry
            constraints = x

            # Perform SMC update step.
            key, *sub_keys = jax.random.split(key, self.num_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, (states, particles, target_weights, weights) = jax.vmap(
                _particle_step, in_axes=(0, None, 0)
            )(sub_keys, constraints, states)

            # Perform resampling.
            total_weight = jax.scipy.special.logsumexp(weights)
            log_normalized_weights = weights - total_weight
            key, sub_key = jax.random.split(key)
            selected_particle_indices = jax.random.categorical(
                sub_key, log_normalized_weights, shape=(self.num_particles,)
            )
            states = states[selected_particle_indices]
            target_weights = target_weights[selected_particle_indices]
            average_weight = total_weight - np.log(self.num_particles)
            weights = jnp.repeat(average_weight, self.num_particles)
            particles = jtu.tree_map(
                lambda v: v[selected_particle_indices], particles
            )
            return (key, states, target_weights, weights), particles

        (key, states, target_weights, weights), particles = jax.lax.scan(
            _inner,
            (key, states, target_weights, weights),
            constraints,
            length=N,
        )

        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        (particles_chm,) = particles.get_retval()
        particles_chm = jtu.tree_map(
            lambda v: jnp.moveaxis(v, (0, 1), (1, 0)), particles_chm
        )
        index_array = np.array([i for i in range(0, N)])
        inaxes_tree = jtu.tree_map(lambda v: 0, particles_chm)
        particles_chm = VectorChoiceMap(
            index_array,
            particles_chm,
        )
        inaxes_tree = VectorChoiceMap(None, inaxes_tree)
        final_target_scores = jax.vmap(
            final_target.importance,
            in_axes=(0, inaxes_tree, None),
        )(sub_keys, particles_chm, ())
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

    # `estimate_logpdf` uses conditional Sequential Monte Carlo (cSMC)
    # to produce an unbiased estimate of the density at retained_choices.
    def estimate_logpdf(self, key, retained_choices, final_target):
        init = self.initial_state(final_target)
        states = jnp.repeat(init, self.num_particles)
        target_weights = jnp.zeros(self.num_particles)
        weights = jnp.zeros(self.num_particles)
        constraints = final_target.constraints
        N = self.num_steps(final_target)

        # Two variants of the particle update step:
        #
        # 1. _particle_step_retained -- operates on the sole
        # propagating particle (convention: this at index 0 of all tensors
        # involved in `estimate_logpdf`)
        #
        # 2. _particle_step_fallthrough -- operates on all the other particles,
        # (required to compute the ultimate density).

        def _particle_step_retained(key, retained_choices, constraints, state):
            new_target = self.step_target(state, constraints, final_target)
            key, (_, particle) = self.step_proposal.importance(
                key, retained_choices, (state, new_target, final_target)
            )
            key, (_, new_target_trace) = new_target.importance(
                key, particle.get_retval(), ()
            )
            target_weight = new_target_trace.get_score()
            weight = new_target_trace.get_score() - particle.get_score()
            new_state = new_target_trace.get_retval()
            return key, (new_state, particle, target_weight, weight)

        def _particle_step_fallthrough(
            key, retained_choices, constraints, state
        ):
            new_target = self.step_target(state, constraints, final_target)
            key, particle = self.step_proposal.simulate(
                key, (state, new_target, final_target)
            )
            key, (_, new_target_trace) = new_target.importance(
                key, particle.get_retval(), ()
            )
            target_weight = new_target_trace.get_score()
            weight = new_target_trace.get_score() - particle.get_score()
            new_state = new_target_trace.get_retval()
            return key, (new_state, particle, target_weight, weight)

        # This switches on the array index value (choosing either to apply the
        # retained computation or the fallthrough computation).
        def _particle_step(key, retained_choices, constraints, state, index):
            check = index == 0
            return jax.lax.cond(
                check,
                _particle_step_retained,
                _particle_step_fallthrough,
                key,
                retained_choices,
                constraints,
                state,
            )

        # Allows `cond`ing on particle index in `jax.vmap`.
        indices = jnp.array([i for i in range(0, self.num_particles)])

        def _inner(carry, x):
            retained_choices, constraints = x
            key, states, target_weights, weights = carry
            key, sub_keys = jax.random.split(key, self.num_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, (particles, target_weights, weights, states) = jax.vmap(
                _particle_step, in_axes=(0, None, 0, 0)
            )(sub_keys, retained_choices, constraints, states, indices)
            total_weight = jax.scipy.special.logsumexp(weights)
            log_normalized_weights = weights - total_weight
            key, sub_key = jax.random.split(key)
            selected_particle_indices = jax.random.categorical(
                sub_key, log_normalized_weights, shape=(self.num_particles)
            )

            # This is a potentially expensive operation in JAX,
            # in interpreter mode -- this will copy the array.
            # However, in JIT mode -- it should be modified to
            # operate in place.
            selected_particle_indices.at[0].set(0)

            target_weights = target_weights[selected_particle_indices]
            average_weight = total_weight - np.log(self.num_particles)
            weights = jnp.repeat(average_weight, self.num_particles)
            states = states[selected_particle_indices]
            return (key, states, target_weights, weights), particles

        (key, states, target_weights, weights), particles = jax.lax.scan(
            _inner,
            (key, states, target_weights, weights),
            (retained_choices, constraints),
            length=N,
        )

        key, sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        final_target_scores = jax.vmap(
            final_target.importance,
            in_axes=(0, 0, None),
        )(sub_keys, particles, ())
        final_weights = weights - target_weights + final_target_scores
        total_weight = jax.scipy.special.logsumexp(final_weights)
        average_weight = total_weight - np.log(self.num_particles)
        return key, final_target_scores[0] - average_weight
