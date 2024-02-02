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
from jax import vmap
from jax.scipy.special import logsumexp

from genjax._src.core.datatypes.generative import Choice, GenerativeFunction, Trace
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
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

#######################
# Particle collection #
#######################


class ParticleCollection(Pytree):
    """A collection of weighted particles.

    Stores the particles (which are `Trace` instances), the log importance weights, the log marginal likelihood estimate, as well as an indicator flag denoting whether the collection is runtime valid or not (`ParticleCollection.is_valid`).
    """

    particles: Trace
    log_weights: FloatArray
    log_marginal_likelihood_estimate: FloatArray
    is_valid: BoolArray

    def get_particles(self):
        return self.particles

    def get_log_weights(self):
        return self.log_weights

    def get_log_marginal_likelihood_estimate(self):
        return self.log_marginal_likelihood_estimate

    def check_valid(self):
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
        particle_collection = algorithm.run_smc(sub_key)
        log_weights = particle_collection.get_log_weights()
        total_weight = logsumexp(log_weights)
        logits = log_weights - total_weight
        idx = categorical.sample(key, logits)
        particle = particle_collection.get_particles().slice(idx)
        log_density_estimate = particle.get_score() - (
            particle_collection.get_log_marginal_likelihood_estimate()
            + total_weight
            - jnp.log(particle_collection.get_num_particles())
        )
        choice = target.project(particle.get_choice())
        return log_density_estimate, choice

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        target: Target,
    ) -> Tuple[FloatArray, Choice]:
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
            - jnp.log(particle_collection.get_num_particles())
        )
        choice = target.project(particle.get_choice())
        return log_density_estimate, choice

    ################
    # VI via GRASP #
    ################

    @typecheck
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ) -> FloatArray:
        pass

    @typecheck
    def estimate_reciprocal_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: Choice,
        w: FloatArray,
    ) -> FloatArray:
        pass


class Initialize(SMCAlgorithm):
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
            jnp.array(0.0),
            jnp.array(True),
        )

    def run_csmc(self, key: PRNGKey, retained: Choice):
        pass


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


########################
# Composite algorithms #
########################


def sampling_importance_resampling(
    model: GenerativeFunction,
    proposal: GenerativeFunction,
    observations: Choice,
    model_args: Tuple,
    proposal_args: Tuple,
):
    pass
