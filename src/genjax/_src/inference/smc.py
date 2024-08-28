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
    Constraint,
    EmptySample,
    GenerativeFunction,
    Retdiff,
    Sample,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.generative.core import GenericProblem
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    BoolArray,
    Callable,
    FloatArray,
    Int,
    PRNGKey,
    typecheck,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    categorical,
)
from genjax._src.generative_functions.static import (
    StaticGenerativeFunction,
    gen,
)
from genjax._src.inference.sp import (
    Algorithm,
    SampleDistribution,
    Target,
)


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


@Pytree.dataclass
class ParticleCollection(Pytree):
    """A collection of weighted particles.

    Stores the particles (which are `Trace` instances), the log importance weights, the log marginal likelihood estimate, as well as an indicator flag denoting whether the collection is runtime valid or not (`ParticleCollection.is_valid`).
    """

    particles: Trace
    log_weights: FloatArray
    is_valid: BoolArray

    def get_particles(self) -> Trace:
        return self.particles

    def get_particle(self, idx) -> Trace:
        return jtu.tree_map(lambda v: v[idx], self.particles)

    def get_log_weights(self) -> FloatArray:
        return self.log_weights

    def get_log_marginal_likelihood_estimate(self) -> FloatArray:
        return logsumexp(self.log_weights) - jnp.log(len(self.log_weights))

    def __getitem__(self, idx) -> tuple[Any, ...]:
        return jtu.tree_map(lambda v: v[idx], (self.particles, self.log_weights))

    def sample_particle(self, key) -> Trace:
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


class SMCAlgorithm(Algorithm):
    """Abstract class for SMC algorithms."""

    @abstractmethod
    def get_num_particles(self):
        raise NotImplementedError

    @abstractmethod
    def get_final_target(self):
        raise NotImplementedError

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
        retained: ChoiceMap,
    ) -> ParticleCollection:
        raise NotImplementedError

    # Convenience method for returning an estimate of the normalizing constant
    # of the target.
    def log_marginal_likelihood_estimate(self, key, target: Target | None = None):
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

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Target,
    ) -> tuple[FloatArray, ChoiceMap]:
        (target,) = args
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

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: ChoiceMap,
        *args: Target,
    ) -> FloatArray:
        (target,) = args
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

    @typecheck
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ) -> FloatArray:
        algorithm = ChangeTarget(self, target)
        key, sub_key = jrandom.split(key)
        particle_collection = algorithm.run_smc(sub_key)
        return particle_collection.get_log_marginal_likelihood_estimate()

    @typecheck
    def estimate_reciprocal_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
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
class Importance(SMCAlgorithm):
    """Accepts as input a `target: Target` and, optionally, a proposal `q: SampleDistribution`.
    `q` should accept a `Target` as input and return a choicemap on a subset
    of the addresses in `target.gen_fn` not in `target.constraints`.

    This initializes a 1-particle `ParticleCollection` by importance sampling from `target` using `q`.

    Any choices in `target.p` not in `q` will be sampled from the internal proposal distribution of `p`,
    given `target.constraints` and the choices sampled by `q`.
    """

    target: Target
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
class ImportanceK(SMCAlgorithm):
    """Given a `target: Target` and a proposal `q: SampleDistribution`, as well as the
    number of particles `k_particles: Int`, initialize a particle collection using
    importance sampling."""

    target: Target
    q: SampleDistribution | None = Pytree.field(default=None)
    k_particles: Int = Pytree.static(default=2)

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


class ResamplingStrategy(Pytree):
    pass


class MultinomialResampling(ResamplingStrategy):
    pass


@Pytree.dataclass
class Resample(Pytree):
    prev: SMCAlgorithm
    resampling_strategy: ResamplingStrategy

    def get_num_particles(self):
        return self.prev.get_num_particles()

    def get_final_target(self):
        return self.prev.get_final_target()

    def run_smc(self, key: PRNGKey):
        pass

    def run_csmc(self, key: PRNGKey, retained: Sample):
        pass


#################
# Change target #
#################


@Pytree.dataclass
class ChangeTarget(SMCAlgorithm):
    prev: SMCAlgorithm
    target: Target

    def get_num_particles(self):
        return self.prev.get_num_particles()

    def get_final_target(self):
        return self.target

    def run_smc(
        self,
        key: PRNGKey,
    ) -> ParticleCollection:
        collection = self.prev.run_smc(key)

        # Convert the existing set of particles and weights
        # to a new set which is properly weighted for the new target.
        def _reweight(key, particle, weight):
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
    ) -> ParticleCollection:
        collection = self.prev.run_csmc(key, retained)

        # Convert the existing set of particles and weights
        # to a new set which is properly weighted for the new target.
        def _reweight(key, particle, weight):
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
    @typecheck
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


########################################################
# Encapsulating SMC moves as re-usable inference logic #
########################################################


@Pytree.dataclass
class KernelTrace(Trace):
    gen_fn: "KernelGenerativeFunction"
    inner: Trace

    def get_args(self) -> tuple[Any, ...]:
        return self.inner.get_args()

    def get_retval(self) -> Any:
        return self.inner.get_retval()

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_score(self) -> FloatArray:
        return self.inner.get_score()

    def get_sample(self) -> Sample:
        return self.inner.get_sample()


@Pytree.dataclass
class KernelGenerativeFunction(GenerativeFunction):
    source: StaticGenerativeFunction

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> Trace:
        gen_fn = gen(self.source)
        tr = gen_fn.simulate(key, args)
        return tr

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[Trace, Weight]:
        raise NotImplementedError

    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
    ) -> tuple[Trace, Weight, Retdiff, UpdateProblem]:
        raise NotImplementedError


@typecheck
def kernel_gen_fn(
    source: Callable[
        [Sample, Target],
        tuple[UpdateProblem, Sample],
    ],
) -> KernelGenerativeFunction:
    return KernelGenerativeFunction(gen(source))


class SMCMove(Pytree):
    @abstractmethod
    def weight_correction(
        self,
        old_latents: Sample,
        new_latents: Sample,
        K_aux: Sample,
        K_aux_score: FloatArray,
    ) -> FloatArray:
        pass


@Pytree.dataclass
class SMCP3Move(SMCMove):
    K: KernelGenerativeFunction
    L: KernelGenerativeFunction

    @typecheck
    def weight_correction(
        self,
        old_latents: Sample,
        new_latents: Sample,
        K_aux: Sample,
        K_aux_score: FloatArray,
    ) -> FloatArray:
        return jnp.array(0.0)


@Pytree.dataclass
class DirectOverload(SMCMove):
    impl: Callable[..., Any]

    @typecheck
    def weight_correction(
        self,
        old_latents: Sample,
        new_latents: Sample,
        K_aux: Sample,
        K_aux_score: FloatArray,
    ) -> FloatArray:
        return jnp.array(0.0)


@Pytree.dataclass
class DeferToInternal(SMCMove):
    @typecheck
    def weight_correction(
        self,
        old_latents: Sample,
        new_latents: Sample,
        K_aux: Sample,
        K_aux_score: FloatArray,
    ) -> FloatArray:
        return jnp.array(0.0)


@Pytree.dataclass
class AttachTrace(Trace):
    gen_fn: "AttachCombinator"
    inner: Trace

    def get_args(self) -> tuple[Any, ...]:
        return self.inner.get_args()

    def get_retval(self) -> Any:
        return self.inner.get_retval()

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn


@Pytree.dataclass
@typecheck
class AttachCombinator(GenerativeFunction):
    gen_fn: GenerativeFunction
    importance_move: SMCMove = Pytree.static(default=DeferToInternal())
    update_move: SMCMove = Pytree.static(default=DeferToInternal())

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> Trace:
        tr = self.gen_fn.simulate(key, args)
        return AttachTrace(self, tr)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[Trace, Weight]:
        move = self.importance_move(constraint)
        match move:
            case SMCP3Move(K, _):
                K_tr = K.simulate(key, (EmptySample(), constraint))
                K_aux_score = K_tr.get_score()
                (new_latents, aux) = K_tr.get_retval()
                w_smc = move.weight_correction(
                    EmptySample(),  # old latents
                    new_latents,  # new latents
                    aux,  # aux from K
                    K_aux_score,
                )
                tr, w = self.gen_fn.importance(key, new_latents, args)
                return tr, w + w_smc

            case DirectOverload(importance_impl):
                return importance_impl(key, constraint, args)

            case DeferToInternal():
                return self.gen_fn.importance(key, constraint, args)

            case _:
                raise Exception("Invalid move type")

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: AttachTrace,
        update_problem: GenericProblem,
    ) -> tuple[Trace, Weight, Retdiff, UpdateProblem]:
        gen_fn_trace = trace.inner
        move = self.update_move(update_problem)
        previous_latents = move.get_previous_latents()
        new_constraint = move.get_new_constraint()
        match move:
            case SMCP3Move(K, _):
                K_tr = K.simulate(key, (previous_latents, new_constraint))
                K_aux_score = K_tr.get_score()
                (new_latents, K_aux) = K_tr.get_retval()
                old_latents = trace.get_sample()
                w_smc = move.weight_correction(
                    old_latents,  # old latents
                    new_latents,  # new latents
                    K_aux,  # aux from K
                    K_aux_score,
                )
                tr, w, retdiff, bwd_problem = self.gen_fn.update(
                    key,
                    gen_fn_trace,
                    GenericProblem(update_problem.argdiffs, new_latents),
                )
                return tr, w + w_smc, retdiff, bwd_problem

            case DirectOverload(update_impl):
                return update_impl(key)

            case DeferToInternal():
                return self.gen_fn.update(key, gen_fn_trace, update_problem)

            case _:
                raise Exception("Invalid move type")


@typecheck
def attach(
    importance_move: SMCMove = DeferToInternal(),
    update_move: SMCMove = DeferToInternal(),
) -> Callable[[GenerativeFunction], AttachCombinator]:
    def decorator(gen_fn) -> AttachCombinator:
        return AttachCombinator(gen_fn, importance_move, update_move)

    return decorator
