# Copyright 2024 MIT Probabilistic Computing Project
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
"""Sequential Monte Carlo ([Chopin & Papaspiliopoulos, 2020](https://link.springer.com/book/10.1007/978-3-030-47845-2), [Del Moral, Doucet, & Jasram 2006](https://academic.oup.com/jrsssb/article/68/3/411/7110641) is an approximate inference framework based on approximating a sequence of target distributions using a weighted collection of particles.

In this module, we provide a set of ingredients for implementing SMC algorithms, including pseudomarginal / recursive auxiliary variants, and variants expressible using SMCP3 ([Lew & Matheos, et al, 2024](https://proceedings.mlr.press/v206/lew23a/lew23a.pdf)) moves.
"""

from abc import abstractmethod

from jax import numpy as jnp
from jax import random as jrandom
from jax import tree_util as jtu
from jax import vmap
from jax.scipy.special import logsumexp

from genjax._src.core.datatypes.generative import Choice, EmptyChoice, Trace
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    ArrayLike,
    BoolArray,
    FloatArray,
    Int,
    PRNGKey,
    Tuple,
    typecheck,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    categorical,
)
from genjax._src.inference.core import ChoiceDistribution, InferenceAlgorithm, Target
from genjax._src.inference.translator import TraceTranslator


# Utility, for CSMC stacking.
@typecheck
def stack_to_first_dim(arr1: ArrayLike, arr2: ArrayLike):
    # Coerce to array, if literal.
    arr1 = jnp.array(arr1, copy=False)
    arr2 = jnp.array(arr2, copy=False)
    # Ensure both arrays are at least 2D
    if arr1.ndim <= 1:
        arr1 = arr1.reshape(-1, 1)
    if arr2.ndim <= 1:
        arr2 = arr2.reshape(-1, 1)

    # Stack the arrays along the first dimension
    result = jnp.concatenate([arr1, arr2], axis=0)
    return jnp.squeeze(result)


#######################
# Particle collection #
#######################


class ParticleCollection(Pytree):
    """A collection of weighted particles.

    Stores the particles (which are `Trace` instances), the log importance weights, the log marginal likelihood estimate, as well as an indicator flag denoting whether the collection is runtime valid or not (`ParticleCollection.is_valid`).
    """

    particles: Trace
    log_weights: FloatArray
    is_valid: BoolArray

    def get_particles(self) -> Trace:
        return self.particles

    def get_log_weights(self) -> FloatArray:
        return self.log_weights

    def get_log_marginal_likelihood_estimate(self) -> FloatArray:
        return logsumexp(self.log_weights) - jnp.log(len(self.log_weights))

    def check_valid(self) -> BoolArray:
        return self.is_valid


####################################
# Abstract type for SMC algorithms #
####################################


class SMCAlgorithm(InferenceAlgorithm):
    """Abstract class for SMC algorithms."""

    @abstractmethod
    def get_num_particles(self):
        raise NotImplementedError

    @abstractmethod
    def get_final_target(self):
        raise NotImplementedError

    @abstractmethod
    def run_smc(self, key: PRNGKey):
        raise NotImplementedError

    @abstractmethod
    def run_csmc(self, key: PRNGKey, retained: Choice):
        raise NotImplementedError

    #########
    # GenSP #
    #########

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        target: Target,
    ) -> Tuple[FloatArray, Choice]:
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        num_particles = self.get_num_particles()
        particle_collection = algorithm.run_smc(sub_key)
        log_weights = particle_collection.get_log_weights()
        total_weight = logsumexp(log_weights)
        logits = log_weights - total_weight
        idx = categorical.sample(key, logits)
        particle = particle_collection.get_particles().slice(idx)
        log_density_estimate = particle.get_score() - (
            particle_collection.get_log_marginal_likelihood_estimate()
            + total_weight
            - jnp.log(num_particles)
        )
        choice = target.project(particle.get_choice())
        return log_density_estimate, choice

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        target: Target,
    ) -> FloatArray:
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        num_particles = self.get_num_particles()
        particle_collection = algorithm.run_csmc(sub_key, latent_choices)
        log_weights = particle_collection.get_log_weights()
        total_weight = logsumexp(log_weights)
        logits = log_weights - total_weight
        idx = categorical.sample(key, logits)
        particle = particle_collection.get_particles().slice(idx)
        log_density_estimate = particle.get_score() - (
            particle_collection.get_log_marginal_likelihood_estimate()
            + total_weight
            - jnp.log(num_particles)
        )
        return log_density_estimate

    ################
    # VI via GRASP #
    ################

    @typecheck
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ) -> FloatArray:
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        particle_collection = algorithm.run_smc(sub_key)
        log_weights = particle_collection.get_log_weights()
        total_weight = logsumexp(log_weights)
        logits = log_weights - total_weight
        idx = categorical.sample(key, logits)
        particle = particle_collection.get_particles().slice(idx)
        log_density_estimate = particle.get_score() - (
            particle_collection.get_log_marginal_likelihood_estimate()
            + total_weight
            - jnp.log(self.get_num_particles())
        )
        return log_density_estimate

    @typecheck
    def estimate_reciprocal_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: Choice,
        w: FloatArray,
    ) -> FloatArray:
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        particle_collection = algorithm.run_csmc(sub_key, latent_choices)
        log_weights = particle_collection.get_log_weights()
        total_weight = logsumexp(log_weights)
        logits = log_weights - total_weight
        idx = categorical.sample(key, logits)
        particle = particle_collection.get_particles().slice(idx)
        log_density_estimate = particle.get_score() - (
            particle_collection.get_log_marginal_likelihood_estimate()
            + total_weight
            - jnp.log(self.get_num_particles())
        )
        return log_density_estimate


@typecheck
class ImportanceSampling(SMCAlgorithm):
    """Given a `target: Target` and a proposal `q: ChoiceDistribution`, as well as the
    number of particles `n_particles: Int`, initialize a particle collection using
    importance sampling."""

    target: Target
    n_particles: Int = Pytree.static()

    def get_num_particles(self):
        return self.n_particles

    def get_final_target(self):
        return self.target

    def run_smc(self, key: PRNGKey):
        key, sub_key = jrandom.split(key)
        sub_keys = jrandom.split(sub_key, self.get_num_particles())
        target_scores, trs = vmap(self.target.generate, in_axes=(0, None))(
            sub_keys, EmptyChoice()
        )
        return ParticleCollection(
            trs,
            target_scores,
            jnp.array(True),
        )

    def run_csmc(self, key: PRNGKey, retained: Choice):
        key, sub_key = jrandom.split(key)
        sub_keys = jrandom.split(sub_key, self.get_num_particles() - 1)
        log_scores, target_traces = vmap(self.target.generate)(sub_keys, EmptyChoice())
        retained_score, retained_trace = self.target.generate(key, retained)
        target_traces = jtu.tree_map(stack_to_first_dim, target_traces, retained_trace)
        target_scores = jtu.tree_map(stack_to_first_dim, log_scores, retained_score)
        return ParticleCollection(
            target_traces,
            target_scores,
            jnp.array(True),
        )


@typecheck
class ProposalImportanceSampling(SMCAlgorithm):
    """Given a `target: Target` and a proposal `q: ChoiceDistribution`, as well as the
    number of particles `n_particles: Int`, initialize a particle collection using
    importance sampling."""

    target: Target
    q: ChoiceDistribution
    n_particles: Int = Pytree.static()

    def get_num_particles(self):
        return self.n_particles

    def get_final_target(self):
        return self.target

    def run_smc(self, key: PRNGKey):
        key, sub_key = jrandom.split(key)
        sub_keys = jrandom.split(sub_key, self.get_num_particles())
        log_weights, choices = vmap(self.q.random_weighted)(sub_keys)
        target_scores, trs = vmap(self.target.generate)(sub_keys, choices)
        return ParticleCollection(
            trs,
            target_scores - log_weights,
            jnp.array(True),
        )

    def run_csmc(self, key: PRNGKey, retained: Choice):
        key, sub_key = jrandom.split(key)
        sub_keys = jrandom.split(sub_key, self.get_num_particles() - 1)
        log_scores, choices = vmap(self.q.random_weighted)(sub_keys)
        retained_choice_score = self.q.estimate_logpdf(key, retained)
        stacked_choices = jtu.tree_map(stack_to_first_dim, choices, retained)
        stacked_scores = jtu.tree_map(
            stack_to_first_dim, log_scores, retained_choice_score
        )
        target_scores, target_traces = vmap(self.target.generate)(
            sub_keys, stacked_choices
        )
        return ParticleCollection(
            target_traces,
            target_scores - stacked_scores,
            jnp.array(True),
        )


##############
# Resampling #
##############


class ResamplingStrategy(Pytree):
    pass


class MultinomialResampling(ResamplingStrategy):
    pass


class Resample:
    prev: SMCAlgorithm
    resampling_strategy: ResamplingStrategy

    def get_num_particles(self):
        return self.prev.get_num_particles()

    def get_final_target(self):
        return self.prev.get_final_target()

    def run_smc(self, key: PRNGKey):
        pass

    def run_csmc(self, key: PRNGKey, retained: Choice):
        pass


#####################
# Trace translation #
#####################


class TraceTranslate(SMCAlgorithm):
    prev: SMCAlgorithm
    translator: TraceTranslator

    def get_num_particles(self):
        return self.prev.get_num_particles()

    def get_final_target(self):
        return self.prev.get_final_target()


#################
# Change target #
#################


class ChangeTarget(SMCAlgorithm):
    prev: SMCAlgorithm
    target: Target

    def get_num_particles(self):
        return self.prev.get_num_particles()

    def get_final_target(self):
        return self.target

    def run_smc(self, key: PRNGKey):
        collection = self.prev.run_smc(key)

        # Convert the existing set of particles and weights
        # to a new set which is properly weighted for the new target.
        def _reweight(key, particle, weight):
            latents = self.prev.get_final_target().project(particle)
            new_score, new_trace = self.target.generate(key, latents)
            this_weight = new_score - particle.get_score() + weight
            return (new_trace, this_weight)

        sub_keys = jrandom.split(key, self.get_num_particles())
        new_particles, new_weights = vmap(_reweight)(
            sub_keys,
            collection.get_particles(),
            collection.get_log_weights(),
        )
        return ParticleCollection(
            new_particles,
            new_weights,
            jnp.array(True),
        )

    def run_csmc(self, key: PRNGKey, retained: Choice):
        collection = self.prev.run_csmc(key, retained)

        # Convert the existing set of particles and weights
        # to a new set which is properly weighted for the new target.
        def _reweight(key, particle, weight):
            latents = self.prev.get_final_target().project(particle)
            new_score, new_trace = self.target.generate(key, latents)
            this_weight = new_score - particle.get_score() + weight
            return (new_trace, this_weight)

        sub_keys = jrandom.split(key, self.get_num_particles())
        new_particles, new_weights = vmap(_reweight)(
            sub_keys,
            collection.get_particles(),
            collection.get_log_weights(),
        )
        return ParticleCollection(
            new_particles,
            new_weights,
            jnp.array(True),
        )
