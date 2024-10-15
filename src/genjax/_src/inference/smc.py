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
"""Sequential Monte Carlo ([Chopin & Papaspiliopoulos, 2020](https://link.springer.com/book/10.1007/978-3-030-47845-2), [Del Moral, Doucet, & Jasram 2006](https://academic.oup.com/jrsssb/article/68/3/411/7110641)) is an approximate inference framework based on approximating a sequence of target distributions using a weighted collection of particles.

In this module, we provide a set of ingredients for implementing SMC algorithms, including pseudomarginal / recursive auxiliary variants, and variants expressible using SMCP3 ([Lew & Matheos, et al, 2024](https://proceedings.mlr.press/v206/lew23a/lew23a.pdf)) moves.
"""

from abc import abstractmethod

from jax import numpy as jnp
from jax import random as jrandom
from jax import tree_util as jtu
from jax import vmap
from jax.scipy.special import logsumexp

from genjax._src.core.generative import (
    ChoiceMap,
    Trace,
)
from genjax._src.core.generative.core import Score
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    BoolArray,
    FloatArray,
    Generic,
    PRNGKey,
    TypeVar,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    categorical,
)
from genjax._src.inference.sp import (
    Algorithm,
    SampleDistribution,
    Target,
)

R = TypeVar("R")

# Utility, for CSMC stacking.


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


@Pytree.dataclass
class ParticleCollection(Generic[R], Pytree):
    """A collection of weighted particles.

    Stores the particles (which are `Trace` instances), the log importance weights, the log marginal likelihood estimate, as well as an indicator flag denoting whether the collection is runtime valid or not (`ParticleCollection.is_valid`).
    """

    particles: Trace[R]
    log_weights: FloatArray
    is_valid: BoolArray

    def get_particles(self) -> Trace[R]:
        return self.particles

    def get_particle(self, idx) -> Trace[R]:
        return jtu.tree_map(lambda v: v[idx], self.particles)

    def get_log_weights(self) -> FloatArray:
        return self.log_weights

    def get_log_marginal_likelihood_estimate(self) -> FloatArray:
        return logsumexp(self.log_weights) - jnp.log(len(self.log_weights))

    def __getitem__(self, idx) -> tuple[Any, ...]:
        return jtu.tree_map(lambda v: v[idx], (self.particles, self.log_weights))

    def sample_particle(self, key) -> Trace[R]:
        """
        Samples a particle from the collection, with probability proportional to its weight.
        """
        log_weights = self.get_log_weights()
        logits = log_weights - logsumexp(log_weights)
        _, idx = categorical.random_weighted(key, logits)
        return self.get_particle(idx)


####################################
# Abstract type for SMC algorithms #
####################################


class SMCAlgorithm(Generic[R], Algorithm[R]):
    """Abstract class for SMC algorithms."""

    @abstractmethod
    def get_num_particles(self) -> int:
        pass

    @abstractmethod
    def get_final_target(self) -> Target[R]:
        pass

    @abstractmethod
    def run_smc(
        self,
        key: PRNGKey,
    ) -> ParticleCollection[R]:
        pass

    @abstractmethod
    def run_csmc(
        self,
        key: PRNGKey,
        retained: ChoiceMap,
    ) -> ParticleCollection[R]:
        pass

    # Convenience method for returning an estimate of the normalizing constant
    # of the target.
    def log_marginal_likelihood_estimate(
        self,
        key: PRNGKey,
        target: Target[R] | None = None,
    ):
        if target:
            algorithm = ChangeTarget(self, target)
        else:
            algorithm = self
        key, sub_key = jrandom.split(key)
        particle_collection = algorithm.run_smc(sub_key)
        return particle_collection.get_log_marginal_likelihood_estimate()

    #########
    # GenSP #
    #########

    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> tuple[Score, ChoiceMap]:
        assert isinstance(args[0], Target)

        target: Target[R] = args[0]
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        particle_collection = algorithm.run_smc(key)
        particle = particle_collection.sample_particle(sub_key)
        log_density_estimate = (
            particle.get_score()
            - particle_collection.get_log_marginal_likelihood_estimate()
        )
        chm = target.filter_to_unconstrained(particle.get_sample())
        return log_density_estimate, chm

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: ChoiceMap,
        *args: tuple[Any, ...],
    ) -> Score:
        assert isinstance(args[0], Target)

        target: Target[R] = args[0]
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        particle_collection = algorithm.run_csmc(key, v)
        particle = particle_collection.sample_particle(sub_key)
        log_density_estimate = (
            particle.get_score()
            - particle_collection.get_log_marginal_likelihood_estimate()
        )
        return log_density_estimate

    ################
    # VI via GRASP #
    ################

    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target[R],
    ) -> FloatArray:
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        particle_collection = algorithm.run_smc(sub_key)
        return particle_collection.get_log_marginal_likelihood_estimate()

    def estimate_reciprocal_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target[R],
        latent_choices: ChoiceMap,
        w: FloatArray,
    ) -> FloatArray:
        algorithm = ChangeTarget(self, target)
        # Special, for ChangeTarget -- to avoid a redundant reweighting step,
        # when we have `w` which (with `latent_choices`) is already properly weighted
        # for the `target`.
        return algorithm.run_csmc_for_normalizing_constant(key, latent_choices, w)


#######################
# Importance sampling #
#######################


@Pytree.dataclass
class Importance(Generic[R], SMCAlgorithm[R]):
    """Accepts as input a `target: Target` and, optionally, a proposal `q: SampleDistribution`.
    `q` should accept a `Target` as input and return a choicemap on a subset
    of the addresses in `target.gen_fn` not in `target.constraints`.

    This initializes a 1-particle `ParticleCollection` by importance sampling from `target` using `q`.

    Any choices in `target.p` not in `q` will be sampled from the internal proposal distribution of `p`,
    given `target.constraints` and the choices sampled by `q`.
    """

    target: Target[R]
    q: SampleDistribution | None = Pytree.field(default=None)

    def get_num_particles(self):
        return 1

    def get_final_target(self):
        return self.target

    def run_smc(self, key: PRNGKey):
        key, sub_key = jrandom.split(key)
        if self.q is not None:
            log_weight, choice = self.q.random_weighted(sub_key, self.target)
            tr, target_score = self.target.importance(key, choice)
        else:
            log_weight = 0.0
            tr, target_score = self.target.importance(key, ChoiceMap.empty())
        return ParticleCollection(
            jtu.tree_map(lambda v: jnp.expand_dims(v, axis=0), tr),
            jnp.array([target_score - log_weight]),
            jnp.array(True),
        )

    def run_csmc(self, key: PRNGKey, retained: ChoiceMap):
        key, sub_key = jrandom.split(key)
        if self.q:
            q_score = self.q.estimate_logpdf(sub_key, retained, self.target)
        else:
            q_score = 0.0
        target_trace, target_score = self.target.importance(key, retained)
        return ParticleCollection(
            jtu.tree_map(lambda v: jnp.expand_dims(v, axis=0), target_trace),
            jnp.array([target_score - q_score]),
            jnp.array(True),
        )


@Pytree.dataclass
class ImportanceK(Generic[R], SMCAlgorithm[R]):
    """Given a `target: Target` and a proposal `q: SampleDistribution`, as well as the
    number of particles `k_particles: int`, initialize a particle collection using
    importance sampling."""

    target: Target[R]
    q: SampleDistribution | None = Pytree.field(default=None)
    k_particles: int = Pytree.static(default=2)

    def get_num_particles(self):
        return self.k_particles

    def get_final_target(self):
        return self.target

    def run_smc(self, key: PRNGKey):
        key, sub_key = jrandom.split(key)
        sub_keys = jrandom.split(sub_key, self.get_num_particles())
        if self.q is not None:
            log_weights, choices = vmap(self.q.random_weighted, in_axes=(0, None))(
                sub_keys, self.target
            )
            trs, target_scores = vmap(self.target.importance)(sub_keys, choices)
        else:
            log_weights = 0.0
            trs, target_scores = vmap(self.target.importance, in_axes=(0, None))(
                sub_keys, ChoiceMap.empty()
            )
        return ParticleCollection(
            trs,
            target_scores - log_weights,
            jnp.array(True),
        )

    def run_csmc(self, key: PRNGKey, retained: ChoiceMap):
        key, sub_key = jrandom.split(key)
        sub_keys = jrandom.split(sub_key, self.get_num_particles() - 1)
        if self.q:
            log_scores, choices = vmap(self.q.random_weighted, in_axes=(0, None))(
                sub_keys, self.target
            )
            retained_choice_score = self.q.estimate_logpdf(key, retained, self.target)
            stacked_choices = jtu.tree_map(stack_to_first_dim, choices, retained)
            stacked_scores = jtu.tree_map(
                stack_to_first_dim, log_scores, retained_choice_score
            )
            sub_keys = jrandom.split(key, self.get_num_particles())
            target_traces, target_scores = vmap(self.target.importance)(
                sub_keys, stacked_choices
            )
        else:
            ignored_traces, ignored_scores = vmap(
                self.target.importance, in_axes=(0, None)
            )(sub_keys, ChoiceMap.empty())
            retained_trace, retained_choice_score = self.target.importance(
                key, retained
            )
            target_scores = jtu.tree_map(
                stack_to_first_dim, ignored_scores, retained_choice_score
            )
            stacked_scores = 0.0
            target_traces = jtu.tree_map(
                stack_to_first_dim, ignored_traces, retained_trace
            )
        return ParticleCollection(
            target_traces,
            target_scores - stacked_scores,
            jnp.array(True),
        )


#################
# Change target #
#################


@Pytree.dataclass
class ChangeTarget(Generic[R], SMCAlgorithm[R]):
    prev: SMCAlgorithm[R]
    target: Target[R]

    def get_num_particles(self):
        return self.prev.get_num_particles()

    def get_final_target(self):
        return self.target

    def run_smc(
        self,
        key: PRNGKey,
    ) -> ParticleCollection[R]:
        collection = self.prev.run_smc(key)

        # Convert the existing set of particles and weights
        # to a new set which is properly weighted for the new target.
        def _reweight(key, particle, weight) -> tuple[Trace[R], Any]:
            latents = self.prev.get_final_target().filter_to_unconstrained(
                particle.get_sample()
            )
            new_trace, new_weight = self.target.importance(key, latents)
            this_weight = new_weight - particle.get_score() + weight
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

    def run_csmc(
        self,
        key: PRNGKey,
        retained: ChoiceMap,
    ) -> ParticleCollection[R]:
        collection = self.prev.run_csmc(key, retained)

        # Convert the existing set of particles and weights
        # to a new set which is properly weighted for the new target.
        def _reweight(key, particle, weight) -> tuple[Trace[R], Any]:
            latents = self.prev.get_final_target().filter_to_unconstrained(
                particle.get_sample()
            )
            new_trace, new_score = self.target.importance(key, latents)
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

    # NOTE: This method is specialized to support the variational inference interface
    # `estimate_reciprocal_normalizing_constant` - by avoiding an extra target
    # reweighting step (which will add extra variance to any derived gradient estimators)
    # It is only available for `ChangeTarget`.

    def run_csmc_for_normalizing_constant(
        self,
        key: PRNGKey,
        latent_choices: ChoiceMap,
        w: FloatArray,
    ) -> FloatArray:
        key, sub_key = jrandom.split(key)
        particle_collection = self.prev.run_csmc(sub_key, latent_choices)

        # Convert the existing set of particles and weights
        # to a new set which is properly weighted for the new target.
        def _reweight(key, particle, weight):
            latents = self.prev.get_final_target().filter_to_unconstrained(
                particle.get_sample()
            )
            _, new_score = self.target.importance(key, latents)
            this_weight = new_score - particle.get_score() + weight
            return this_weight

        num_particles = self.get_num_particles()
        sub_keys = jrandom.split(key, num_particles - 1)
        new_rejected_weights = vmap(_reweight)(
            sub_keys,
            jtu.tree_map(lambda v: v[:-1], particle_collection.get_particles()),
            jtu.tree_map(lambda v: v[:-1], particle_collection.get_log_weights()),
        )
        retained_score = particle_collection.get_particle(-1).get_score()
        retained_weight = particle_collection.get_log_weights()[-1]
        all_weights = stack_to_first_dim(
            new_rejected_weights,
            w - retained_score + retained_weight,
        )
        total_weight = logsumexp(all_weights)
        return retained_score - (total_weight - jnp.log(num_particles))
