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

import abc
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import ValueChoiceMap
from genjax.core.pytree import Pytree
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target


#####
# Utilities
#####


@dataclass
class ParticleCollection(Pytree):
    particles: Any
    weights: Any
    lml_est: Any

    def flatten(self):
        return (self.particles, self.weights, self.lml_est), ()

    def log_marginal_likelihood(self):
        return (
            self.lml_est
            + jax.scipy.special.logsumexp(self.weights)
            - jnp.log(len(self.particles))
        )


def effective_sample_size():
    raise AssertionError()


#####
# Abstract types for SMC DSL
#####


@dataclass
class SMCAlgorithm(Pytree):
    @abc.abstractmethod
    def num_particles(self) -> int:
        pass

    @abc.abstractmethod
    def final_target(self) -> Target:
        pass

    @abc.abstractmethod
    def run_smc(self, key) -> Tuple[Any, ParticleCollection]:
        pass

    @abc.abstractmethod
    def run_csmc(self, key, choices) -> Tuple[Any, ParticleCollection]:
        pass

    def random_weighted(self, key, target):
        algorithm = SMCChangeTarget(self, target)
        key, particle_collection = algorithm.run_smc(key)
        weights = particle_collection.weights
        total_weight = jax.scipy.special.logsumexp(weights)
        log_normalized_weights = weights - total_weight
        key, sub_key = jax.random.split(key)
        particle_index = jax.random.categorical(
            sub_key, log_normalized_weights
        )
        particle = jtu.tree_map(
            lambda v: v[particle_index], particle_collection.particles
        )
        chm = particle.get_choices()
        score = (
            particle_collection.lml_est
            + total_weight
            - jnp.log(len(particle_collection.particles))
        )
        return key, (score, chm)

    def estimate_logpdf(self, key, choices, target):
        algorithm = SMCChangeTarget(self, target)
        key, collection = algorithm.run_csmc(key, choices)
        retained = jtu.tree_map(lambda v: v[-1], collection.particles)
        score = retained.get_score() - collection.log_marginal_likelihood()
        return key, (score, retained.get_choices())


#####
# Ingredients
#####


@dataclass
class SMCInit(SMCAlgorithm):
    q: Any
    target: Target
    n_particles: int

    def flatten(self):
        return (), (self.q, self.target, self.num_particles)

    def num_particles(self):
        return self.n_particles

    def run_smc(self, key):
        key, *sub_keys = jax.random.split(key, self.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, proposals = jax.vmap(self.q.simulate, in_axes=(0, None))(
            sub_keys, self.target
        )
        constraints = self.target.constraints
        constraints_axes_tree = jtu.tree_map(lambda v: None, constraints)
        (chm,) = proposals.get_retval()
        chm_axes_tree = jtu.tree_map(lambda v: 0, chm)
        merged = chm.merge(constraints)
        merged_axes_tree = chm_axes_tree.merge(constraints_axes_tree)
        key, *sub_keys = jax.random.split(key, self.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (_, traces) = jax.vmap(
            self.target.p, in_axes=(0, merged_axes_tree, None)
        )(sub_keys, merged, self.target.args)
        weights = traces.get_score() - proposals.get_score()
        return key, ParticleCollection(traces, weights, 0.0)

    def run_csmc(self, key, choices):
        key, *sub_keys = jax.random.split(key, self.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, proposals = jax.vmap(self.q.simulate, in_axes=(0, None))(
            sub_keys, self.target
        )

        ################################################
        # In JIT, should be optimized to in-place.

        # Set retained.
        key, kept = self.q.importance(
            key, ValueChoiceMap(choices), (self.target,)
        )

        def _mutate_in_place(v, kept):
            new = v.at[-1].set(kept)
            return new

        proposals = jtu.tree_map(_mutate_in_place, proposals, kept)
        ################################################

        constraints = self.target.constraints
        constraints_axes_tree = jtu.tree_map(lambda v: None, constraints)
        (chm,) = proposals.get_retval()
        chm_axes_tree = jtu.tree_map(lambda v: 0, chm)
        merged = chm.merge(constraints)
        merged_axes_tree = chm_axes_tree.merge(constraints_axes_tree)
        key, *sub_keys = jax.random.split(key, self.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (_, traces) = jax.vmap(
            self.target.p, in_axes=(0, merged_axes_tree, None)
        )(sub_keys, merged, self.target.args)
        weights = traces.get_score() - proposals.get_score()
        return key, ParticleCollection(
            traces,
            weights,
            0.0,
        )


@dataclass
class SMCExtend(SMCAlgorithm):
    previous: SMCAlgorithm
    k: ProxDistribution
    new_args: Tuple
    new_constraints: ChoiceMap

    def flatten(self):
        return (self.new_args, self.new_constraints), (self.previous, self.k)

    def num_particles(self):
        return self.previous.num_particles()

    def run_smc(self, key):
        new_target = self.final_target()
        key, collection = self.previous.run_smc(key)
        key, *sub_keys = jax.random.split(key, self.num_particles() + 1)
        sub_keys = jnp.array(sub_keys)
        extension = jax.vmap(self.k.simulate, in_axes=(0, (0, None)))(
            sub_keys, (collection.particles, new_target)
        )
        (particle_chm,) = extension.get_retval()
        key, *sub_keys = jax.random.split(key, self.num_particles() + 1)
        sub_keys = jnp.array(sub_keys)
        _, (_, new_target_trace) = jax.vmap(
            new_target.importance, in_axes=(0, 0)
        )(sub_keys, particle_chm, ())
        target_score = new_target_trace.get_score()
        weights = target_score - extension.get_score()
        particles = jtu.tree_map(
            lambda v1, v2: jnp.hstack((v1, v2)),
            collection.particles,
            extension,
        )
        return key, ParticleCollection(
            particles,
            collection.weights + weights,
            collection.lml_est,
        )

    def run_csmc(self, key, choices):
        new_target = self.final_target()
        merged = new_target.constraints.merge(choices)
        key, new_retained = new_target.p.importance(
            key, merged, new_target.args
        )


@dataclass
class SMCChangeTarget(SMCAlgorithm):
    previous: SMCAlgorithm
    new_target: Target

    def flatten(self):
        return (), (self.previous, self.target)

    def num_particles(self):
        return self.previous.num_particles()

    def final_target(self):
        return self.new_target

    def run_smc(self, key):
        key, collection = self.previous.run_smc(key)
        old_target_latents = self.previous.final_target().latent_selection()

        def _inner(key, particle, weight):
            latents, _ = old_target_latents.filter(particle.get_choices())
            merged = self.new_target.constraints.merge(latents)
            key, (_, new_trace) = self.new_target.p.importance(
                key, merged, self.new_target.args
            )
            weight = new_trace.get_score() - particle.get_score() + weight
            return particle, weight

        key, *sub_keys = jax.random.split(
            key, self.previous.num_particles() + 1
        )
        sub_keys = jnp.array(sub_keys)
        particles, weights = jax.vmap(_inner, in_axes=(0, 0, 0))(
            sub_keys, collection.particles, collection.weights
        )
        return key, ParticleCollection(
            particles,
            weights,
            collection.lml_est,
        )

    def run_csmc(self, key, choices):
        key, collection = self.previous.run_csmc(key, choices)
        old_target_latents = self.previous.final_target().latent_selection()

        def _inner(key, particle, weight):
            latents, _ = old_target_latents.filter(particle.get_choices())
            merged = self.new_target.constraints.merge(latents)
            key, (_, new_trace) = self.new_target.p.importance(
                key, merged, self.new_target.args
            )
            weight = new_trace.get_score() - particle.get_score() + weight
            return particle, weight

        key, *sub_keys = jax.random.split(
            key, self.previous.num_particles() + 1
        )
        sub_keys = jnp.array(sub_keys)
        particles, weights = jax.vmap(_inner, in_axes=(0, 0, 0))(
            sub_keys, collection.particles, collection.weights
        )
        return key, ParticleCollection(
            particles,
            weights,
            collection.lml_est,
        )


@dataclass
class SMCRejuvenate(SMCAlgorithm):
    previous: SMCAlgorithm
    kernel: Callable

    def flatten(self):
        return (), (self.previous, self.kernel)

    def num_particles(self):
        return self.previous.num_particles()

    def final_target(self):
        return self.previous.final_target()


@dataclass
class SMCClone(SMCAlgorithm):
    previous: SMCAlgorithm
    factor: int

    def flatten(self):
        return (), (self.previous, self.factor)

    def num_particles(self):
        return self.previous.num_particles() * self.factor

    def final_target(self):
        return self.previous.final_target()

    def run_smc(self, key):
        key, collection = self.previous.run_smc(key)
        return key, jtu.tree_map(
            lambda v: jnp.repeat(v, self.factor), collection
        )

    def run_csmc(self, key, choices):
        key, collection = self.previous.run_csmc(key, choices)
        return key, jtu.tree_map(
            lambda v: jnp.repeat(v, self.factor), collection
        )


@dataclass
class SMCResample(SMCAlgorithm):
    previous: SMCAlgorithm
    ess_threshold: float
    how_many: int

    def flatten(self):
        return (), (self.previous, self.ess, self.how_many)

    def final_target(self):
        return self.previous.final_target()

    def num_particles(self):
        return self.how_many

    def run_smc(self, key):
        key, collection = self.previous.run_smc(key)
        num_particles = len(collection.particles)
        total_weight = jax.scipy.special.logsumexp(collection.weights)
        log_normalized_weights = collection.weights - total_weight
        _ = effective_sample_size(log_normalized_weights) > self.ess_threshold

        key, sub_key = jax.random.split(key)
        selected_particle_indices = jax.random.categorical(
            sub_key, log_normalized_weights, shape=(self.how_many,)
        )
        particles = jtu.tree_map(
            lambda v: v[selected_particle_indices], collection.particles
        )
        weights = jnp.zeros(self.how_many)
        avg_weight = total_weight - jnp.log(num_particles)
        return key, ParticleCollection(
            particles,
            weights,
            avg_weight + collection.lml_est,
        )

    def run_csmc(self, key, choices):
        key, collection = self.previous.run_csmc(key, choices)
        num_particles = len(collection.particles)
        total_weight = jax.scipy.special.logsumexp(collection.weights)
        log_normalized_weights = collection.weights - total_weight
        # ess_check = (
        #    effective_sample_size(log_normalized_weights) > self.ess_threshold
        # )

        key, sub_key = jax.random.split(key)
        selected_particle_indices = jax.random.categorical(
            sub_key, log_normalized_weights, shape=(self.how_many,)
        )

        ################################################
        # In JIT, should be optimized to in-place.

        # Set retained.
        selected_particle_indices = selected_particle_indices.at[-1].set(
            num_particles
        )
        ################################################

        particles = jtu.tree_map(
            lambda v: v[selected_particle_indices], collection.particles
        )
        weights = jnp.zeros(self.how_many)
        avg_weight = total_weight - jnp.log(num_particles)
        return key, ParticleCollection(
            particles,
            weights,
            avg_weight + collection.lml_est,
        )


@dataclass
class SMCChain(SMCAlgorithm):
    previous: SMCAlgorithm
    sequence: Sequence[SMCAlgorithm]
    unfold_lengths: Callable[[Target], int]
    target: Target
