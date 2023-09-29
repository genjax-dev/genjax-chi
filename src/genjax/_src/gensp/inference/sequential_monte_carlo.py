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
"""This module contains a mini-language for constructing compositional SMC
algorithms.

It includes two main components: `SMCPropagator` and `SMCAlgorithm`.

`SMCPropagator` ingredients accept a set of inputs which they expect to
be given by higher-level `SMCAlgorithm` combinators. By exposing their
requirements to their callers, they can be chained together to support
higher-level patterns with efficient JAX compilation.

`SMCAlgorithm` ingredients are self-contained SMC algorithm instances
which can be run to produce particle populations which are properly
weighted with respect to their registered inference `Target` instances.
"""

import abc
import dataclasses
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.special import logsumexp

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.pytree import Pytree
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import Float
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.gensp.choice_map_distribution import ChoiceMapDistribution
from genjax._src.gensp.target import Target


#####
# Utilities
#####


@dataclasses.dataclass
class ParticleCollection(Pytree):
    particles: Any
    weights: Any
    lml_est: Any

    def flatten(self):
        return (self.particles, self.weights, self.lml_est), ()

    def get_particles(self):
        return self.particles

    def get_weights(self):
        return self.weights

    def log_marginal_likelihood(self):
        return (
            self.lml_est
            + jax.scipy.special.logsumexp(self.weights)
            - jnp.log(len(self.particles))
        )

    def effective_sample_size(self):
        total_weight = logsumexp(self.weights)
        log_normalized_weights = self.weights - total_weight
        log_ess = -logsumexp(2.0 * log_normalized_weights)
        return jnp.exp(log_ess)

    def __getitem__(self, indices):
        new_particles, new_weights = jtu.tree_map(
            lambda v: v[indices], (self.particles, self.weights)
        )
        return ParticleCollection(new_particles, new_weights, self.lml_est)

    def concat(self, new_particle, new_weight):
        new_particles, new_weights = jtu.tree_map(
            lambda v1, v2: jnp.concatenate((v1, v2), axis=-1),
            (self.particles, self.weights),
            (new_particle, new_weight),
        )
        return ParticleCollection(new_particles, new_weights, self.lml_est)


@dataclasses.dataclass
class SMCState(Pytree):
    collection: ParticleCollection
    target: Target
    num_particles: Int

    def flatten(self):
        return (
            self.collection,
            self.target,
        ), (self.num_particles,)

    def get_collection(self):
        return self.collection

    def get_target(self):
        return self.target

    def get_num_particles(self):
        return self.num_particles

    def get_particles(self):
        return self.collection.get_particles()

    def get_weights(self):
        return self.collection.get_weights()


#################
# SMCPropagator #
#################


@dataclasses.dataclass
class SMCPropagator(Pytree):
    @abc.abstractmethod
    def propagate_target(self, target: Target, *args):
        pass

    @abc.abstractmethod
    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
        *args,
    ) -> Tuple[PRNGKey, SMCState]:
        pass

    @abc.abstractmethod
    def pullback(
        self,
        key: PRNGKey,
        retained: ChoiceMap,
        *args,
    ) -> Tuple[PRNGKey, ChoiceMap, Callable]:
        pass

    # Essentially monadic bind. Take a propagator,
    # and compose it with the existing algorithm.
    def and_then(self, propagator: "SMCPropagator"):
        return SMCSequencePropagator.new(self, propagator)


#####
# Propagator ingredients
#####

# Propagators operate on `state: SMCState` and transform it
# potentially changing the target, changing the collection of particles,# etc.
#
# They are different from `SMCAlgorithm` below -- because they
# require a `state: SMCState` to run their methods on.
#
# You can pair propagators with `SMCInit` (importance sampling to get an
# initial state) below. `SMCInit` is like a monadic lift. The propagators
# define functionality (and compositional functions) which are
# compatible in a monad-like DSL.


@dataclasses.dataclass
class SMCExtendPropagator(SMCPropagator):
    k: ChoiceMapDistribution

    def flatten(self):
        return (), (self.k,)

    def propagate_target(self, target: Target, new_args: Tuple, new_choices: ChoiceMap):
        old_constraints = target.constraints
        new_constraints = old_constraints.safe_merge(new_choices)
        return Target(
            target.p,
            target.choice_map_coercion,
            new_args,
            new_constraints,
        )

    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
        argdiffs: Tuple,
        new_choices: ChoiceMap,
    ) -> Tuple[PRNGKey, SMCState]:
        new_args = tree_diff_primal(argdiffs)
        new_target = self.propagate_target(
            state.get_target(),
            new_args,
            new_choices,
        )
        key, *sub_keys = jax.random.split(key, state.get_num_particles() + 1)
        sub_keys = jnp.array(sub_keys)
        _, extension = jax.vmap(self.k.simulate, in_axes=(0, (0, None)))(
            sub_keys, (state.get_particles(), new_target)
        )
        particle_chm = extension.get_retval()
        k_score = extension.get_score()
        key, *sub_keys = jax.random.split(key, state.get_num_particles() + 1)
        sub_keys = jnp.array(sub_keys)
        _, (model_score_change, new_target_trace, _) = jax.vmap(
            new_target.p.update, in_axes=(0, 0, None, None)
        )(sub_keys, state.collection.particles, particle_chm, new_target.args)
        weight_change = -k_score + model_score_change
        return key, SMCState(
            ParticleCollection(
                new_target_trace,
                state.collection.weights + weight_change,
                state.collection.lml_est,
            ),
            new_target,
            state.get_num_particles(),
        )

    def _apply_conditional(
        self,
        key: PRNGKey,
        state: SMCState,
        argdiffs: Tuple,
        new_choices: ChoiceMap,
        new_retained_trace: Trace,
        forward_weight_retained: Float,
    ):
        new_args = tree_diff_primal(argdiffs)
        new_target = self.propagate_target(
            state.get_target(),
            new_args,
            new_choices,
        )
        num_particles = state.get_num_particles()
        key, *sub_keys = jax.random.split(key, num_particles)
        sub_keys = jnp.array(sub_keys)
        sliced_collection = state.collection[0 : num_particles - 1]
        _, extension = jax.vmap(self.k.simulate, in_axes=(0, (0, None)))(
            sub_keys, (sliced_collection.particles, new_target)
        )
        particle_chm = extension.get_retval()
        k_score = extension.get_score()
        key, *sub_keys = jax.random.split(key, num_particles)
        sub_keys = jnp.array(sub_keys)
        _, (_, model_score_change, new_particles, _) = jax.vmap(
            new_target.p.update, in_axes=(0, 0, None, None)
        )(
            sub_keys,
            sliced_collection.particles,
            particle_chm,
            new_target.args,
        )
        new_weights = sliced_collection.get_weights() - k_score + model_score_change
        new_collection = ParticleCollection(
            new_particles, new_weights, state.collection.lml_est
        )

        final_collection = new_collection.concat(
            new_retained_trace, forward_weight_retained
        )

        return key, SMCState(
            final_collection,
            new_target,
            state.get_num_particles(),
        )

    def pullback(
        self,
        key: PRNGKey,
        previous_target: Target,
        retained: ChoiceMap,
        argdiffs: Tuple,
        new_constraints: ChoiceMap,
    ) -> Tuple[PRNGKey, SMCState]:
        new_args = tree_diff_primal(argdiffs)
        new_target = self.propagate_target(
            previous_target,
            new_args,
            new_constraints,
        )
        merged = new_target.constraints.safe_merge(retained)
        key, (_, new_retained_trace) = new_target.p.importance(
            key, merged, new_target.args
        )
        key, (_, _, previous_trace, discard) = new_retained_trace.update(
            key, EmptyChoiceMap(), argdiffs
        )
        previous_latents = retained.filter(discard.get_selection().complement())
        key, (_, forward_weight_retained) = self.k.assess(
            key,
            ValueChoiceMap(discard),
            (previous_trace, new_target),
        )

        def _pullback(key, state, argdiffs, new_choices):
            return self._apply_conditional(
                key,
                state,
                argdiffs,
                new_choices,
                new_retained_trace,
                forward_weight_retained,
            )

        return key, previous_latents, _pullback


@dataclasses.dataclass
class SMCChangeTargetPropagator(SMCPropagator):
    new_target: Target

    def flatten(self):
        return (self.new_target,), ()

    def propagate_target(self, _: Target):
        return self.new_target

    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
    ) -> Tuple[PRNGKey, SMCState]:
        old_target_latents_selection = state.get_target().latent_selection()
        key, *sub_keys = jax.random.split(key, state.get_num_particles() + 1)
        old_target_latents = state.get_particles().filter(old_target_latents_selection)

        def _importance(key, latents):
            merged = self.new_target.constraints.safe_merge(latents)
            key, (_, new_particle) = self.new_target.p.importance(
                key, merged, self.new_target.args
            )
            return key, new_particle

        _, new_particles = jax.vmap(_importance, in_axes=(0, 0))(
            sub_keys, old_target_latents
        )
        weights = (
            new_particles.get_score()
            - state.get_particles().get_score()
            + state.get_weights()
        )
        final_collection = ParticleCollection(
            new_particles, weights, state.collection.lml_est
        )
        return key, SMCState(
            final_collection,
            self.new_target,
            state.get_num_particles(),
        )

    def pullback(
        self,
        key: PRNGKey,
        retained: ChoiceMap,
    ) -> Tuple[PRNGKey, ChoiceMap, Callable]:
        return key, retained, self.apply


@dataclasses.dataclass
class SMCSequencePropagator(SMCPropagator):
    sequence: Sequence[SMCPropagator]

    def flatten(self):
        return (self.sequence,)

    @classmethod
    def new(fst: SMCPropagator, snd: SMCPropagator):
        if isinstance(fst, SMCSequencePropagator) and isinstance(
            snd, SMCSequencePropagator
        ):
            return SMCSequencePropagator(
                [*fst.sequence, *snd.sequence],
            )
        elif isinstance(fst, SMCSequencePropagator):
            return SMCSequencePropagator(
                [*fst.sequence, snd],
            )
        elif isinstance(snd, SMCSequencePropagator):
            return SMCSequencePropagator(
                [fst, *snd.sequence],
            )
        else:
            return SMCSequencePropagator([fst, snd])

    def propagate_target(self, target: Target, args_sequence: Sequence[Tuple]):
        for (propagator, args) in zip(self.sequence, args_sequence):
            target = propagator.propagate_target(target, *args)
        return target

    def apply(
        self,
        key: PRNGKey,
        state: SMCState,
        args_sequence: Sequence[Tuple],
    ) -> Tuple[PRNGKey, SMCState]:
        for (propagator, args) in zip(self.sequence, args_sequence):
            key, state = propagator.apply(key, state, *args)
        return key, state

    def pullback(
        self,
        key: PRNGKey,
        retained: ChoiceMap,
    ) -> Tuple[PRNGKey, ChoiceMap, Callable]:
        pass


################
# SMCAlgorithm #
################


@dataclasses.dataclass
class SMCAlgorithm(ChoiceMapDistribution):
    @abc.abstractmethod
    def get_final_target(self) -> Target:
        pass

    @abc.abstractmethod
    def run_smc(
        self,
        key: PRNGKey,
    ) -> Tuple[PRNGKey, SMCState]:
        pass

    @abc.abstractmethod
    def run_csmc(
        self,
        key: PRNGKey,
        choices: ChoiceMap,
    ) -> Tuple[PRNGKey, SMCState]:
        pass

    # Essentially monadic bind. Take a propagator,
    # and compose it with the existing algorithm.
    def and_then(self, propagator: SMCPropagator, *args):
        return SMCCompose(self, propagator, args)

    def random_weighted(self, key, target):
        algorithm = SMCChangeTargetPropagator(target)
        key, state = self.run_smc(key, target)
        key, state = algorithm.propagate(key, state)
        particle_collection = state.collection
        weights = particle_collection.weights
        total_weight = jax.scipy.special.logsumexp(weights)
        log_normalized_weights = weights - total_weight
        key, sub_key = jax.random.split(key)
        particle_index = jax.random.categorical(sub_key, log_normalized_weights)
        particle = jtu.tree_map(
            lambda v: v[particle_index], particle_collection.particles
        )
        chm = particle.get_choices()
        score = (
            particle_collection.lml_est + total_weight - jnp.log(state.num_particles)
        )
        return key, (score, chm)

    def estimate_logpdf(self, key, choices, target):
        algorithm = SMCChangeTargetPropagator(target)
        key, state = algorithm.run_csmc(key, target, choices)
        collection = state.get_collection()
        retained = jtu.tree_map(lambda v: v[-1], collection.particles)
        score = retained.get_score() - collection.log_marginal_likelihood()
        return key, (score, retained.get_choices())


@dataclasses.dataclass
class SMCInit(SMCAlgorithm):
    q: Any
    num_particles: Int
    target: Target

    def flatten(self):
        return (), (self.q, self.num_particles)

    def get_final_target(self):
        return self.target

    def run_smc(self, key: PRNGKey, target: Target) -> Tuple[PRNGKey, SMCState]:
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, proposals = jax.vmap(self.q.simulate, in_axes=(0, None))(sub_keys, (target,))
        chm = proposals.get_retval()
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (_, traces) = jax.vmap(target.importance, in_axes=(0, 0, None))(
            sub_keys, chm, target.args
        )
        weights = traces.get_score() - proposals.get_score()
        return key, SMCState(
            ParticleCollection(traces, weights, 0.0),
            target,
            self.num_particles,
        )

    def run_csmc(self, key, choices, target):
        key, *sub_keys = jax.random.split(key, self.num_particles)
        sub_keys = jnp.array(sub_keys)
        _, proposals = jax.vmap(self.q.simulate, in_axes=(0, None))(sub_keys, target)

        # Set retained.
        key, kept = self.q.importance(
            key,
            ValueChoiceMap.new(choices),
            (target,),
        )

        proposals = jtu.tree_map(lambda v1, v2: jnp.hstack((v1, v2)), proposals, kept)

        chm = proposals.get_retval()
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (_, traces) = jax.vmap(target.importance, in_axes=(0, None, None))(
            sub_keys, chm, target.args
        )
        weights = traces.get_score() - proposals.get_score()
        return key, SMCState(
            ParticleCollection(traces, weights, 0.0),
            target,
            self.num_particles,
        )


#####
# Combinators
#####


@dataclasses.dataclass
class SMCCompose(SMCAlgorithm):
    prev: SMCAlgorithm
    propagator: SMCPropagator
    propagator_args: Tuple

    def flatten(self):
        return (self.prev, self.propagator, self.propagator_args), ()

    def get_final_target(self):
        initial_target = self.prev.get_final_target()
        final_target = self.propagator.propagate_target(
            initial_target, *self.propagator_args
        )
        return final_target

    def run_smc(self, key):
        key, state = self.prev.run_smc(key)
        key, state = self.propagator.apply(key, state, *self.propagator_args)
        return key, state

    def run_csmc(self, key, choices):
        old_target = self.prev.get_final_target()
        key, retained, forward = self.propagator.pullback(
            key, old_target, choices, *self.propagator_args
        )
        key, state = self.prev.run_csmc(key, retained)
        key, state = forward(key, state, *self.propagator_args)
        return key, state


##############
# Shorthands #
##############

# Primitive algorithms + propagators.
Init = SMCInit
Extend = SMCExtendPropagator
ChangeTarget = SMCChangeTargetPropagator

# Higher-level combinators.
Compose = SMCCompose
Sequence = SMCSequencePropagator
