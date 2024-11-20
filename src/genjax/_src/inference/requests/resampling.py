
##############
# Resampling #
##############

import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
from jax import grad
from jax.lax import scan, cond
from tensorflow_probability.substrates import jax as tfp
from jax.scipy.special import logsumexp

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    EditRequest,
    Retdiff,
    Score,
    Selection,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.generative.core import EditRequest
from genjax._src.core.generative.requests import DiffAnnotate
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    BoolArray,
    IntArray,
    PRNGKey,
    static_check_supports_grad,
)

@Pytree.dataclass
class ESSResamplingStrategy(EditRequest):
    """Abstract class for Resampling strategies that also resample based on the Effective Sample Size (ESS) of the particles.
    Args:
        - `ess_threshold`: The ESS threshold at which to resample.
    Returns:
        A batched trace that maybe performed the resampling.
    """
    ess_threshold: FloatArray = Pytree.field(
        default=Pytree.const(1.0))

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

def change_score(
    tr: Trace[Any],
    score: Score
    ) -> Trace[Any]:



# TODO: I feel like this shouldn't work with JAX/jit as the conditional cannot be compiled away at trace time.
# TODO: might be wrong due to shape change for 3 reasons:
# - discussion with Xuan about similar algorithm in Gen.jl
# - because of weight diff computation
@Pytree.dataclass
class MultinomialResampling(ESSResamplingStrategy):
    """Performs multinomial resampling (i.e. simple random resampling) of the particles in the filter. Each trace (i.e. particle) is resampled with probability proportional to its weight."""
    ess_threshold: FloatArray = Pytree.field(
        default=Pytree.const(1.0))  # a value of 1 will always resample

    def edit(
        self,
        key: PRNGKey,
        tr: Trace[Any],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[Any], Weight, Retdiff[Any], "EditRequest"]:
        particles = tr.get_choices()
        log_weights = tr.get_score()
        ess = compute_log_ess(log_weights)
        num_particles = tr.get_score().shape

        def resample(key):
            key, key2 = jrand.split(key)
            keys = jrand.split(key, num_particles)
            idxs = jrand.categorical(keys, log_weights)
            #TODO: wrong, categorical takes only one key and if the shape is non-vector tensor it doesn't work.
            new_particles = jtu.tree_map(lambda v: v[idxs], particles)
            avg_weight = logsumexp(log_weights, axis=log_weights.shape) - log_weights.size
            new_weights = avg_weight + jnp.zeros(num_particles)
            new_trace, _, retdiff, _ = Update(new_particles).edit(key2, tr, argdiffs) #TODO: need to edit new_trace to have new_weights as score
            return new_trace, new_weights - log_weights, retdiff, MultinomialResampling(self.ess_threshold)

        def no_resample():
            return tr, jnp.zeros(num_particles), Diff.no_change(tr.get_retval), MultinomialResampling(self.ess_threshold)

        return cond(ess < jnp.log(self.ess_threshold), lambda key: resample(key), lambda _: no_resample(), key)

@Pytree.dataclass
class SystematicResampling(ESSResamplingStrategy):
    """Perform systematic resampling of the particles in the filter, which reduces variance relative to multinomial sampling. Look at the cumulative sum of the normalized weights, and then pick the particles that are closest to the strata. This is a deterministic resampling scheme that is more efficient than multinomial resampling.

    Extra arg:
    - `sort_particles = jnp.array(False)`: Set to `True` to sort particles by weight before stratification.
    """
    ess_threshold: FloatArray = Pytree.field(
        default=Pytree.const(1.0))  # a value of 1 will always resample

    # TODO: need to add sorting logic
    # Optionally sort particles by weight before resampling
    # order = jnp.argsort(normalized_log_weights) if self.sort_particles else jnp.arange(len(normalized_log_weights))
    sort_particles: BoolArray = Pytree.field(default=Pytree.const(False))

    def run_smc(self, key: PRNGKey):
        collection = self.prev.run_smc(key)
        log_weights = collection.get_log_weights()
        ess = compute_log_ess(log_weights)
        if ess < jnp.log(self.ess_threshold):
            normalized_log_weights = log_weights - logsumexp(log_weights)
            # Assumes that their sum is not zero.
            u0 = jrand.uniform(key)
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
            keys = jrand.split(key, self.how_many)
            us = jrand.uniform(keys)
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
            keys = jrand.split(key, n_resampled)
            idx2 = jrand.categorical(keys, resample_weights)
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
