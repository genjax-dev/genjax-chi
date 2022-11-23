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
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import ValueChoiceMap
from genjax.core.pytree import Pytree
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target
from genjax.inference.kernels.kernel import MCMCKernel


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
class SMCState(Pytree):
    retained: Union[None, ChoiceMap]
    collection: ParticleCollection
    target: Target
    n_particles: int

    def flatten(self):
        return (
            self.retained,
            self.collection,
            self.target,
        ), (self.n_particles,)


@dataclass
class SMCPropagator(Pytree):
    def __call__(self, key, state, *args):
        if state.retained is None:
            return self.propagate(key, state, *args)
        else:
            return self.conditional_propagate(key, state, *args)

    @abc.abstractmethod
    def propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
        *args,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        pass

    @abc.abstractmethod
    def conditional_propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
        *args,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        pass


#####
# Propagator ingredients
#####

# Propagators operate on `state: SMCState` and transform it
# potentially changing the target, changing the collection of particles,# etc.
#
# They are different from `SMCAlgorithm` below -- because they
# require a `state: SMCState` to run their methods on.
#
# You can pair propagators with `Init` (importance sampling to get an
# initial state) below. `Init` is like a monadic lift. The propagators
# define functionality (and compositional functions) which are
# compatible in a monad-like DSL.


@dataclass
class ChangeTarget(SMCPropagator):
    new_target: Target

    def flatten(self):
        return (), (self.new_target)

    def propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        old_target_latents = state.target.latent_selection()
        n_particles = state.n_particles
        particles = state.collection.particles

        def _inner(key, particle, weight):
            latents, _ = old_target_latents.filter(particle.get_choices())
            latents = latents.strip()
            key, (_, new_trace) = self.new_target.importance(
                key,
                latents,
                (),
            )
            weight = new_trace.get_score() - particle.get_score() + weight
            return particle, weight

        key, *sub_keys = jax.random.split(key, n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        particles, weights = jax.vmap(_inner, in_axes=(0, 0, 0))(
            sub_keys,
            particles,
            state.collection.weights,
        )
        return key, SMCState(
            None,
            ParticleCollection(
                particles,
                weights,
                state.collection.lml_est,
            ),
            self.new_target,
            n_particles,
        )

    def conditional_propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        assert state.retained is not None
        return self.propagate(key, state)


@dataclass
class Extend(SMCPropagator):
    k: ProxDistribution

    def flatten(self):
        return (), (self.k,)

    def propagate_target(self, target, new_args, new_choices):
        old_constraints = target.constraints
        new_constraints = old_constraints.merge(new_choices)
        return Target(
            target.p,
            target.choice_map_coercion,
            new_args,
            new_constraints,
        )

    def propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
        new_args: Tuple,
        new_choices: ChoiceMap,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        new_target = self.propagate_target(
            state.target,
            new_args,
            new_choices,
        )
        key, *sub_keys = jax.random.split(key, state.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, extension = jax.vmap(self.k.simulate, in_axes=(0, (0, None)))(
            sub_keys, (state.collection.particles, new_target)
        )
        (particle_chm,) = extension.get_retval()
        k_score = extension.get_score()
        key, *sub_keys = jax.random.split(key, state.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (model_score_change, new_target_trace, _) = jax.vmap(
            new_target.p.update, in_axes=(0, 0, None, None)
        )(sub_keys, state.collection.particles, particle_chm, new_target.args)
        weight_change = -k_score + model_score_change
        return key, SMCState(
            None,
            ParticleCollection(
                new_target_trace,
                state.collection.weights + weight_change,
                state.collection.lml_est,
            ),
            new_target,
            state.n_particles,
        )

    def conditional_propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
        new_args: Tuple,
        new_constraints: ChoiceMap,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        assert state.retained is not None
        old_target = state.target
        new_target = self.propagate_target(
            old_target,
            new_args,
            new_constraints,
        )
        retained = state.retained
        key, (_, new_retained_trace) = new_target.importance(key, retained, ())
        gen_fn = new_retained_trace.get_gen_fn()
        key, (previous_trace, _, discard) = gen_fn.update(
            key, new_retained_trace, EmptyChoiceMap(), old_target.args()
        )
        previous_latents, _ = (
            discard.get_selection().complement().filter(retained)
        )
        key, (forward_weight_retained, _) = self.k.importance(
            key,
            ValueChoiceMap.new(discard),
            (previous_trace, new_target),
        )

        def conditional_pullback(key, state):
            key, *sub_keys = jax.random.split(key, state.n_particles)
            sub_keys = jnp.array(sub_keys)
            _, extension = jax.vmap(self.k.simulate, in_axes=(0, (0, None)))(
                sub_keys, (state.collection.particles, new_target)
            )
            (particle_chm,) = extension.get_retval()
            k_score = extension.get_score()
            key, *sub_keys = jax.random.split(key, state.n_particles)
            sub_keys = jnp.array(sub_keys)
            _, (model_score_change, new_target_trace, _) = jax.vmap(
                new_target.p.update, in_axes=(0, 0, None, None)
            )(
                sub_keys,
                state.collection.particles,
                particle_chm,
                new_target.args,
            )
            weight_change = -k_score + model_score_change
            non_retained_weights = (
                state.collection.weights[0 : state.n_particles] + weight_change
            )
            particles = jtu.tree_map(
                lambda v1, v2: jnp.hstack((v1, v2)),
                new_target_trace,
                new_retained_trace,
            )
            weights = jnp.hstack(
                (
                    non_retained_weights,
                    state.collection.weights[-1]
                    + forward_weight_retained
                    + new_retained_trace.get_score()
                    - state.collection.particles[-1].get_score(),
                )
            )

            return key, SMCState(
                new_retained_trace.strip(),
                ParticleCollection(
                    particles, weights, state.collection.lml_est
                ),
                new_target,
                state.n_particles,
            )

        return previous_latents, conditional_pullback


@dataclass
class Resample(SMCPropagator):
    ess_threshold: float
    how_many: int

    def flatten(self):
        return (), (self.ess_threshold, self.how_many)

    def propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        n_particles = state.n_particles
        total_weight = jax.scipy.special.logsumexp(state.collection.weights)
        log_normalized_weights = state.collection.weights - total_weight
        key, sub_key = jax.random.split(key)
        selected_particle_indices = jax.random.categorical(
            sub_key, log_normalized_weights, shape=(self.how_many,)
        )
        particles = jtu.tree_map(
            lambda v: v[selected_particle_indices], state.collection.particles
        )
        weights = jnp.zeros(self.how_many)
        avg_weight = total_weight - jnp.log(n_particles)
        return key, SMCState(
            None,
            ParticleCollection(
                particles,
                weights,
                avg_weight + state.collection.lml_est,
            ),
            state.target,
            n_particles,
        )

    def conditional_propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        n_particles = state.n_particles
        total_weight = jax.scipy.special.logsumexp(state.collection.weights)
        log_normalized_weights = state.collection.weights - total_weight
        key, sub_key = jax.random.split(key)
        selected_particle_indices = jax.random.categorical(
            sub_key, log_normalized_weights, shape=(self.how_many,)
        )

        ################################################
        # In JIT, should be optimized to in-place.

        # Set retained.
        selected_particle_indices = selected_particle_indices.at[-1].set(
            n_particles
        )
        ################################################

        particles = jtu.tree_map(
            lambda v: v[selected_particle_indices], state.collection.particles
        )
        weights = jnp.zeros(self.how_many)
        avg_weight = total_weight - jnp.log(n_particles)
        return key, SMCState(
            None,
            ParticleCollection(
                particles,
                weights,
                avg_weight + state.collection.lml_est,
            ),
            state.target,
            n_particles,
        )


@dataclass
class Rejuvenate(SMCPropagator):
    kernel: MCMCKernel
    kernel_args: Tuple

    def flatten(self):
        return (self.kernel_args,), (self.kernel,)

    def propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        key, *sub_keys = jax.random.split(key, state.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        key, (new_particles, _) = jax.vmap(self.kernel, in_axes=(0, 0, None))(
            sub_keys, state.collection.particles, self.kernel_args
        )
        return key, SMCState(
            None,
            ParticleCollection(
                new_particles,
                state.collection.weights,
                state.collection.lml_est,
            ),
            state.target,
            state.n_particles,
        )

    def conditional_propagate(
        self,
        key: jax.random.PRNGKey,
        state: SMCState,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        pass


#####
# Algorithm DSL
#####


@dataclass
class SMCAlgorithm(ProxDistribution):
    @abc.abstractmethod
    def run_smc(
        self,
        key: jax.random.PRNGKey,
        target: Target,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        pass

    @abc.abstractmethod
    def run_csmc(
        self,
        key: jax.random.PRNGKey,
        choices: ChoiceMap,
        target: Target,
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        pass

    # Essentially monadic bind. Take a propagator,
    # and compose it with the existing algorithm.
    def and_then(self, propagator: SMCPropagator, *args):
        return Compose(self, propagator, args)

    def random_weighted(self, key, target):
        algorithm = ChangeTarget(target)
        key, state = self.run_smc(key, target)
        key, state = algorithm.propagate(key, state)
        particle_collection = state.collection
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
            - jnp.log(state.n_particles)
        )
        return key, (score, chm)

    def estimate_logpdf(self, key, choices, target):
        algorithm = ChangeTarget(target)
        key, state = self.run_csmc(key, target, choices)
        key, state = algorithm.conditional_propagate(key, state)
        collection = state.collection
        retained = jtu.tree_map(lambda v: v[-1], collection.particles)
        score = retained.get_score() - collection.log_marginal_likelihood()
        return key, (score, retained.get_choices())


@dataclass
class Init(SMCAlgorithm):
    q: Any
    n_particles: int

    def flatten(self):
        return (), (self.q, self.n_particles)

    def run_smc(
        self, key: jax.random.PRNGKey, target: Target
    ) -> Tuple[jax.random.PRNGKey, SMCState]:
        key, *sub_keys = jax.random.split(key, self.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, proposals = jax.vmap(self.q.simulate, in_axes=(0, None))(
            sub_keys, (target,)
        )
        (chm,) = proposals.get_retval()
        key, *sub_keys = jax.random.split(key, self.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (_, traces) = jax.vmap(target.importance, in_axes=(0, 0, None))(
            sub_keys, chm, target.args
        )
        weights = traces.get_score() - proposals.get_score()
        return key, SMCState(
            None,
            ParticleCollection(traces, weights, 0.0),
            target,
            self.n_particles,
        )

    def run_csmc(self, key, choices, target):
        key, *sub_keys = jax.random.split(key, self.n_particles)
        sub_keys = jnp.array(sub_keys)
        _, proposals = jax.vmap(self.q.simulate, in_axes=(0, None))(
            sub_keys, target
        )

        # Set retained.
        key, kept = self.q.importance(
            key,
            ValueChoiceMap.new(choices),
            (target,),
        )

        proposals = jtu.tree_map(
            lambda v1, v2: jnp.hstack((v1, v2)), proposals, kept
        )

        (chm,) = proposals.get_retval()
        key, *sub_keys = jax.random.split(key, self.n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (_, traces) = jax.vmap(target.importance, in_axes=(0, None, None))(
            sub_keys, chm, target.args
        )
        weights = traces.get_score() - proposals.get_score()
        return key, SMCState(
            kept.strip(),
            ParticleCollection(traces, weights, 0.0),
            target,
            self.n_particles,
        )


@dataclass
class Compose(SMCAlgorithm):
    prev: SMCAlgorithm
    propagator: SMCPropagator
    propagator_args: Tuple

    def flatten(self):
        return (self.prev, self.propagator, self.propagator_args), ()

    def run_smc(self, key, target):
        key, state = self.prev.run_smc(key, target)
        key, state = self.propagator.propagate(
            key, state, *self.propagator_args
        )
        return key, state

    def run_csmc(self, key, choices, target):
        key, state = self.prev.run_csmc(key, choices, target)
        key, state = self.propagator.conditional_propagate(
            key, state, *self.propagator_args
        )
        return key, state
