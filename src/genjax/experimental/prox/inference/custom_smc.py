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

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax.core.datatypes import ValueChoiceMap
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target


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
        target_scores = jnp.zeros(self.num_particles)
        weights = jnp.zeros(self.num_particles)
        constraints = final_target.constraints
        N = self.num_steps(final_target)

        def _particle_step(key, constraints, state):
            new_target = self.step_target(state, constraints, final_target)
            key, particle = self.step_proposal.simulate(
                key, (state, new_target)
            )
            (particle_chm,) = particle.get_retval()
            key, (_, new_target_trace) = new_target.importance(
                key, particle_chm, ()
            )
            target_score = new_target_trace.get_score()
            weight = new_target_trace.get_score() - particle.get_score()
            (new_state,) = new_target_trace.get_retval()
            return key, (new_state, particle, target_score, weight)

        def _inner(carry, x):
            key, sub_keys, states, prev_target_scores, prev_weights = carry
            constraints = x

            sub_keys, (states, particles, target_scores, weights) = jax.vmap(
                _particle_step, in_axes=(0, None, 0)
            )(sub_keys, constraints, states)
            target_scores = prev_target_scores + target_scores
            weights = prev_weights + weights

            # Compute resampling indices and perform resampling on
            # weights, scores, etc -- we defer resampling the particles
            # themselves until later.
            total_weight = jax.scipy.special.logsumexp(weights)
            log_normalized_weights = weights - total_weight
            key, sub_key = jax.random.split(key)
            selected_particle_indices = jax.random.categorical(
                sub_key,
                log_normalized_weights,
                shape=(self.num_particles,),
            )
            states = states[selected_particle_indices]
            target_scores = target_scores[selected_particle_indices]
            average_weight = total_weight - np.log(self.num_particles)
            weights = jnp.repeat(average_weight, self.num_particles)
            particles = jtu.tree_map(
                lambda v: v[selected_particle_indices], particles
            )
            return (key, sub_keys, states, target_scores, weights), (
                selected_particle_indices,
                particles,
            )

        # Preallocate sub-keys to carry through the scan call.
        key, *sub_keys = jax.random.split(
            key,
            self.num_particles + 1,
        )
        sub_keys = jnp.array(sub_keys)

        (key, sub_keys, states, target_scores, weights), (
            selected_particle_indices,
            particles,
        ) = jax.lax.scan(
            _inner,
            (key, sub_keys, states, target_scores, weights),
            constraints,
            length=N,
        )

        # Here, we begin to prepare `particles` for the final target score
        # evaluation. scan stacks the particles on the last (the 1) axis
        # so we swap to the front so we can `vmap` over that.
        particles = jtu.tree_map(lambda v: jnp.swapaxes(v, 0, 1), particles)
        selected_particle_indices = jtu.tree_map(
            lambda v: jnp.swapaxes(v, 0, 1), selected_particle_indices
        )
        (particles_chm,) = particles.get_retval()

        # Here, we need to perform resampling (which we did not do inside of scan)
        # before we score particles with the final target.
        def apply_resample_indices(i, particles):
            r = jnp.arange(selected_particle_indices.shape[1])
            ind = selected_particle_indices[:, i]
            return jtu.tree_map(
                lambda data: jnp.where(r >= i, data, data[ind]), particles
            )

        particles_chm = jax.lax.fori_loop(
            0,
            selected_particle_indices.shape[1],
            apply_resample_indices,
            particles_chm,
        )

        # Now, we prepare for a `vmap` call to final target
        # importance -- by building an axes tree to tell `vmap`
        # where to broadcast.
        inaxes_tree = jtu.tree_map(lambda v: 0, particles_chm)
        sub_keys, (final_target_scores, final_tr) = jax.vmap(
            final_target.importance,
            in_axes=(0, inaxes_tree, None),
        )(sub_keys, particles_chm, ())

        # Compute the final weights and perform the last
        # resampling step (to select our final chosen particle).
        final_weights = weights - target_scores + final_target_scores
        total_weight = jax.scipy.special.logsumexp(final_weights)
        average_weight = total_weight - np.log(self.num_particles)
        log_normalized_weights = final_weights - total_weight
        key, sub_key = jax.random.split(key)
        selected_particle_index = jax.random.categorical(
            key, log_normalized_weights
        )

        # `final_tr` comes from `final_target.importance`
        # instead of just returning a particle choice map,
        # we allow `final_target` to coerce the constraints
        # into the shape of the choice map which it returns.
        selected_particle = jtu.tree_map(
            lambda v: v[selected_particle_index],
            final_target.get_latents(final_tr),
        )

        return key, (
            final_target_scores[selected_particle_index] - average_weight,
            selected_particle,
        )

    # `estimate_logpdf` uses conditional Sequential Monte Carlo (cSMC)
    # to produce an unbiased estimate of the density at retained_choices.
    def estimate_logpdf(self, key, retained_choices, final_target):
        init = self.initial_state(final_target)
        states = jnp.repeat(init, self.num_particles)
        target_scores = jnp.zeros(self.num_particles)
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

        def _particle_step_retained(
            key,
            new_target,
            retained_choices,
            constraints,
            state,
        ):
            retained_choices = retained_choices.get_choices()
            retained_choices = ValueChoiceMap.new(retained_choices)
            key, (_, particle) = self.step_proposal.importance(
                key, retained_choices, (state, new_target)
            )
            return key, particle

        def _particle_step_fallthrough(
            key,
            new_target,
            retained_choices,
            constraints,
            state,
        ):
            key, particle = self.step_proposal.simulate(
                key, (state, new_target)
            )
            return key, particle

        # This switches on the array index value
        # (choosing either to apply the retained computation
        # or the fallthrough computation).
        def _particle_step(key, retained_choices, constraints, state, index):
            check = index == 0
            new_target = self.step_target(state, constraints, final_target)
            key, particle = jax.lax.cond(
                check,
                _particle_step_retained,
                _particle_step_fallthrough,
                key,
                new_target,
                retained_choices,
                constraints,
                state,
            )
            (particle_chm,) = particle.get_retval()
            key, (_, new_target_trace) = new_target.importance(
                key, particle_chm, ()
            )
            target_score = new_target_trace.get_score()
            weight = new_target_trace.get_score() - particle.get_score()
            (new_state,) = new_target_trace.get_retval()
            return key, (new_state, particle, target_score, weight)

        # Allows `cond`ing on particle index in `jax.vmap`.
        indices = jnp.array([i for i in range(0, self.num_particles)])

        def _inner(carry, x):
            key, sub_keys, states, prev_target_scores, prev_weights = carry
            retained_choices, constraints = x
            sub_keys, (states, particles, target_scores, weights) = jax.vmap(
                _particle_step, in_axes=(0, None, None, 0, 0)
            )(sub_keys, retained_choices, constraints, states, indices)

            target_scores += prev_target_scores
            weights += prev_weights

            # Constrained resampling.
            total_weight = jax.scipy.special.logsumexp(weights)
            log_normalized_weights = weights - total_weight
            key, sub_key = jax.random.split(key)
            selected_particle_indices = jax.random.categorical(
                sub_key, log_normalized_weights, shape=(self.num_particles,)
            )

            # This is a potentially expensive operation in JAX,
            # in interpreter mode -- this will copy the array.
            # However, in JIT mode -- it should be modified to
            # operate in place.
            fixed = selected_particle_indices.at[0].set(0)

            target_scores = target_scores[fixed]
            average_weight = total_weight - np.log(self.num_particles)
            weights = jnp.repeat(average_weight, self.num_particles)
            states = states[fixed]
            particles = jtu.tree_map(lambda v: v[fixed], particles)
            return (key, sub_keys, states, target_scores, weights), (
                fixed,
                particles,
            )

        # We pre-allocate keys which we carry through.
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)

        # Here is the main scan loop --
        # performing SMC update steps, tabulating weights,
        # target scores, and returning out particles
        # as well as the ancestor indices.
        (key, sub_keys, states, target_scores, weights), (
            selected_particle_indices,
            particles,
        ) = jax.lax.scan(
            _inner,
            (key, sub_keys, states, target_scores, weights),
            (retained_choices, constraints),
            length=N,
        )

        # Here, we begin to prepare `particles` for the final target score
        # evaluation. scan stacks the particles on the last (the 1) axis
        # so we swap to the front so we can `vmap` over that.
        particles = jtu.tree_map(lambda v: jnp.swapaxes(v, 0, 1), particles)
        selected_particle_indices = jtu.tree_map(
            lambda v: jnp.swapaxes(v, 0, 1), selected_particle_indices
        )
        (particles_chm,) = particles.get_retval()

        # Here, we need to perform resampling (which we did not do inside of scan)
        # before we score particles with the final target.
        def apply_resample_indices(i, particles):
            r = jnp.arange(selected_particle_indices.shape[1])
            ind = selected_particle_indices[:, i]
            return jtu.tree_map(
                lambda data: jnp.where(r >= i, data, data[ind]), particles
            )

        particles_chm = jax.lax.fori_loop(
            0,
            selected_particle_indices.shape[1],
            apply_resample_indices,
            particles_chm,
        )

        # Now, we prepare for a `vmap` call to final target
        # importance -- by building an axes tree to tell `vmap`
        # where to broadcast.
        inaxes_tree = jtu.tree_map(lambda v: 0, particles_chm)
        sub_keys, (final_target_scores, _) = jax.vmap(
            final_target.importance,
            in_axes=(0, inaxes_tree, None),
        )(sub_keys, particles_chm, ())

        # Compute the final weights.
        final_weights = weights - target_scores + final_target_scores
        total_weight = jax.scipy.special.logsumexp(final_weights)
        average_weight = total_weight - np.log(self.num_particles)

        return key, (
            final_target_scores[0] - average_weight,
            retained_choices,
        )
