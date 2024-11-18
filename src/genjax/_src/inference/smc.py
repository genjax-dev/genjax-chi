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

import jaxtyping as jtyping

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


##############
# Resampling #
##############


# TODO: we will want a lazy resampling version to avoid unnecessary computation.
# In particular, the problem comes from the fact that we do not have an efficient data structure for lazy duplication of particles as in the Julia implementation. As the copies will often be sparse (e.g. because of Masking), we can quickly run into memory and performance issues. One solution is to record the indices of the particles that need to be duplicated, and then duplicate them when needed. In general it may be that we need actual particles at every step and then we may need some other tricks, but for some cases we can get away with lazy duplication, where we only perform the construction at the very end.
# TODO: compared to the Julia version in GenParticleFilters, doesn't implement
# the `priority_fn` nor the `check` to throw an error for invalid normalized
# weights (all NaNs or zeros).
# Also doesn't use the marginal loglikelihood estimate for stability of the weights of long SMC chains which can underflow.
class ESSResamplingStrategy(SMCAlgorithm):
    """Abstract class for Resampling strategies that also resample based on the Effective Sample Size (ESS) of the particles.
    Args:
        - `prev`: The previous SMCAlgorithm.
        - `how_many`: The number of particles to resample.
        - `ess_threshold`: The ESS threshold at which to resample.

    Returns:
        ParticleCollection: a resampled particle collection.
    """

    prev: SMCAlgorithm
    how_many: Int
    ess_threshold: Optional[FloatArray] = Pytree.field(
        default=Pytree.const(1.0)
    )  # a value of 1 will always resample

    def get_num_particles(self):
        return self.how_many

    def get_final_target(self):
        return self.prev.get_final_target()

    @abstractmethod
    def run_smc(
        self,
        key: PRNGKey,
    ) -> ParticleCollection:
        raise NotImplementedError

    @abstractmethod
    def run_csmc(
        self,
        key: PRNGKey,
        retained: Sample,
    ) -> ParticleCollection:
        raise NotImplementedError


def compute_log_ess(log_weights):
    """Compute the log of the Effective Sample Size (ESS) of a set of log unnormalized weights."""
    return 2 * logsumexp(log_weights) - logsumexp(2 * log_weights)


def update_weights(log_weights, log_priorities):
    """Update particle weights after a resampling step."""
    n = len(log_weights)
    # If priorities aren't customized, set all log weights to 0
    if log_weights == log_priorities:
        return jnp.zeros(n)
    # Otherwise, set new weights to the ratio of weights over priorities
    log_ws = log_weights - log_priorities
    # Adjust new weights such that they sum to the number of particles
    return log_ws - logsumexp(log_ws) + jnp.log(n)


@Pytree.dataclass
class MultinomialResampling(ESSResamplingStrategy):
    """Performs multinomial resampling (i.e. simple random resampling) of the particles in the filter. Each trace (i.e. particle) is resampled with probability proportional to its weight."""

    # TODO: how the heck do I make these inherited from the abstract parent class ESSResamplingStrategy?
    prev: SMCAlgorithm
    how_many: Int
    ess_threshold: Optional[FloatArray] = Pytree.field(
        default=Pytree.const(1.0)
    )  # a value of 1 will always resample

    def run_smc(self, key: PRNGKey):
        collection = self.prev.run_smc(key)
        log_weights = collection.get_log_weights()
        ess = compute_log_ess(log_weights)
        if ess < jnp.log(self.ess_threshold):
            keys = jrandom.split(key, self.how_many)
            idxs = jrandom.categorical(keys, log_weights)
            new_particles = jtu.tree_map(lambda v: v[idxs], collection.get_particles())
            new_weights = jnp.zeros(self.how_many)
            return ParticleCollection(new_particles, new_weights, jnp.array(True))
        return collection

    def run_csmc(
        self,
        key: PRNGKey,
        retained: Sample,
    ) -> ParticleCollection:
        raise NotImplementedError


@Pytree.dataclass
class SystematicResampling(ESSResamplingStrategy):
    """Perform systematic resampling of the particles in the filter, which reduces variance relative to multinomial sampling. Look at the cumulative sum of the normalized weights, and then pick the particles that are closest to the strata. This is a deterministic resampling scheme that is more efficient than multinomial resampling.

    Extra arg:
    - `sort_particles = jnp.array(False)`: Set to `True` to sort particles by weight before stratification.
    """

    prev: SMCAlgorithm
    how_many: Int
    ess_threshold: Optional[FloatArray] = Pytree.field(
        default=Pytree.const(1.0)
    )  # a value of 1 will always resample
    # TODO: need to add sorting logic
    # Optionally sort particles by weight before resampling
    # order = jnp.argsort(normalized_log_weights) if self.sort_particles else jnp.arange(len(normalized_log_weights))
    sort_particles: Optional[BoolArray] = Pytree.field(default=Pytree.const(False))

    def run_smc(self, key: PRNGKey):
        collection = self.prev.run_smc(key)
        log_weights = collection.get_log_weights()
        ess = compute_log_ess(log_weights)
        if ess < jnp.log(self.ess_threshold):
            normalized_log_weights = log_weights - logsumexp(log_weights)
            # Assumes that their sum is not zero.
            u0 = jrandom.uniform(key)
            c = jnp.cumsum(normalized_log_weights)
            u = (jnp.arange(self.how_many) + u0) / self.how_many
            idxs = jnp.searchsorted(c, u, side="right")
            # TODO: is that efficient?
            new_particles = jtu.tree_map(lambda v: v[idxs], collection.get_particles())
            # TODO: should that be 1 or average of the weights or 1/N?
            new_weights = jnp.zeros(self.how_many)
            return ParticleCollection(new_particles, new_weights, jnp.array(True))
        return collection

    def run_csmc(
        self,
        key: PRNGKey,
        retained: Sample,
    ) -> ParticleCollection:
        raise NotImplementedError


@Pytree.dataclass
class StratifiedResampling(ESSResamplingStrategy):
    """Performs stratified resampling of the particles in the filter, which reduces variance relative to multinomial sampling.
    First, uniform random samples ``u_1, ..., u_n`` are drawn within the strata ``[0, 1/n)``, ..., ``[n-1/n, 1)``, where ``n`` is the number of particles. Then, given the cumulative normalized weights ``W_k = Σ_{j=1}^{k} w_j ``, sample the ``k``th particle for each ``u_i`` where ``W_{k-1} ≤ u_i < W_k``.

    Extra arg:
     - sort_particles = jnp.array(False). Set to `True` to sort particles by weight before stratification.
    """

    prev: SMCAlgorithm
    how_many: jtyping.Int
    ess_threshold: Optional[FloatArray] = Pytree.field(
        default=Pytree.const(1.0)
    )  # a value of 1 will always resample
    sort_particles: Optional[BoolArray] = Pytree.field(default=Pytree.const(False))

    def run_smc(self, key: PRNGKey):
        collection = self.prev.run_smc(key)
        log_weights = collection.get_log_weights()
        ess = compute_log_ess(log_weights)
        if ess < jnp.log(self.ess_threshold):
            normalized_log_weights = log_weights - logsumexp(log_weights)
            # Assumes that their sum is not zero.
            keys = jrandom.split(key, self.how_many)
            us = jrandom.uniform(keys)
            c = jnp.cumsum(normalized_log_weights)
            u = (jnp.arange(self.how_many) + us) / self.how_many
            idxs = jnp.searchsorted(c, u, side="right")
            new_particles = jtu.tree_map(lambda v: v[idxs], collection.get_particles())
            new_weights = jnp.zeros(self.how_many)
            return ParticleCollection(new_particles, new_weights, jnp.array(True))
        return collection

    def run_csmc(
        self,
        key: PRNGKey,
        retained: Sample,
    ) -> ParticleCollection:
        raise NotImplementedError


@Pytree.dataclass
class ResidualResampling(ESSResamplingStrategy):
    """Performs residual resampling of the particles in the filter, which reduces variance relative to multinomial sampling. For each particle with normalized weight ``w_i``, ``⌊n w_i⌋`` copies are resampled, where ``n`` is the total number of particles. The remainder are sampled with probability proportional to ``n w_i - ⌊n w_i⌋`` for each particle ``i``."""

    prev: SMCAlgorithm
    how_many: Int
    ess_threshold: Optional[FloatArray] = Pytree.field(
        default=Pytree.const(1.0)
    )  # a value of 1 will always resample

    def run_smc(self, key: PRNGKey):
        collection = self.prev.run_smc(key)
        log_weights = collection.get_log_weights()
        ess = compute_log_ess(log_weights)
        if ess < jnp.log(self.ess_threshold):
            normalized_log_weights = log_weights - logsumexp(log_weights)
            # Assumes that their sum is not zero.

            # Deterministically copy previous particles according to their weights
            n_copies_per_weight = jnp.floor(self.how_many * normalized_log_weights)
            n_copies = jnp.sum(n_copies_per_weight)
            n_resampled = self.how_many - n_copies
            # JAX may be unhappy with dynamic values here
            idx1 = jnp.repeat(jnp.arange(self.how_many), n_copies_per_weight)

            # Sample remainder according to residual weights
            # TODO: check if works when all are 0.
            resample_weights = (
                self.how_many * normalized_log_weights - n_copies_per_weight
            )
            keys = jrandom.split(key, n_resampled)
            idx2 = jrandom.categorical(keys, resample_weights)
            idxs = jnp.concatenate([idx1, idx2])
            new_particles = jtu.tree_map(lambda v: v[idxs], collection.get_particles())
            new_weights = jnp.zeros(self.how_many)
            return ParticleCollection(new_particles, new_weights, jnp.array(True))
        return collection

    def run_csmc(
        self,
        key: PRNGKey,
        retained: Sample,
    ) -> ParticleCollection:
        raise NotImplementedError


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
