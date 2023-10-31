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

import adevjax
import jax
import jax.numpy as jnp
from adevjax import ADEVPrimitive
from adevjax import flip_enum
from adevjax import geometric_reinforce
from adevjax import mv_normal_diag_reparam
from adevjax import mv_normal_reparam
from adevjax import normal_reinforce
from adevjax import normal_reparam
from adevjax import sample_with_key

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.utilities import tree_stack
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.distribution import ExactDensity
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_bernoulli,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_categorical,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_geometric,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_mv_normal,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_mv_normal_diag,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_normal,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_uniform,
)
from genjax._src.gensp.sp_distribution import SPDistribution
from genjax._src.gensp.target import Target
from genjax._src.gensp.target import target


##########################################
# Differentiable distribution primitives #
##########################################


@dataclass
class ADEVDistribution(ExactDensity):
    differentiable_logpdf: Callable
    adev_primitive: ADEVPrimitive

    def flatten(self):
        return (self.adev_primitive,), (self.differentiable_logpdf,)

    @classmethod
    def new(cls, adev_prim, diff_logpdf):
        return ADEVDistribution(diff_logpdf, adev_prim)

    def sample(self, key, *args):
        return sample_with_key(self.adev_primitive, key, *args)

    def logpdf(self, v, *args):
        return self.differentiable_logpdf(v, *args)


flip_enum = ADEVDistribution.new(
    adevjax.flip_enum,
    lambda v, logits: tfp_bernoulli.logpdf(v, logits=logits),
)

flip_reinforce = ADEVDistribution.new(
    adevjax.flip_reinforce,
    lambda v, p: tfp_bernoulli.logpdf(v, probs=p),
)

categorical_enum = ADEVDistribution.new(
    adevjax.categorical_enum_parallel,
    lambda v, p: tfp_categorical.logpdf(v, probs=p),
)

normal_reinforce = ADEVDistribution.new(
    adevjax.normal_reinforce,
    lambda v, μ, σ: tfp_normal.logpdf(v, μ, σ),
)

normal_reparam = ADEVDistribution.new(
    adevjax.normal_reparam,
    lambda v, μ, σ: tfp_normal.logpdf(v, μ, σ),
)

mv_normal_reparam = ADEVDistribution.new(
    adevjax.mv_normal_reparam,
    lambda v, μ, Σ: tfp_mv_normal.logpdf(v, μ, Σ),
)

mv_normal_diag_reparam = ADEVDistribution.new(
    adevjax.mv_normal_diag_reparam,
    lambda v, μ, Σ_diag: tfp_mv_normal_diag.logpdf(v, μ, Σ_diag),
)

geometric_reinforce = ADEVDistribution.new(
    adevjax.geometric_reinforce,
    lambda v, *args: tfp_geometric.logpdf(v, *args),
)

uniform = ADEVDistribution.new(
    adevjax.uniform,
    lambda v: tfp_uniform.logpdf(v, 0.0, 1.0),
)

#######################################
# Differentiable inference primitives #
#######################################


@dataclass
class SPAlgorithm(Pytree):
    @abc.abstractmethod
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ):
        pass

    @abc.abstractmethod
    def estimate_recip_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: ChoiceMap,
        w: FloatArray,
    ):
        pass


@dataclass
class DefaultSIR(SPAlgorithm):
    num_particles: Int

    def flatten(self):
        return (), (self.num_particles,)

    @typecheck
    def estimate_recip_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: ChoiceMap,
        w: FloatArray,
    ):
        # Kludge.
        log_weights = []
        for i in range(0, self.num_particles - 1):
            key, sub_key = jax.random.split(key)
            (_, ps) = target.importance(sub_key)
            log_weights.append(ps.get_score())

        log_weights.append(w)

        lws = tree_stack(log_weights)
        # END Kludge.

        tw = jax.scipy.special.logsumexp(lws)
        aw = tw - jnp.log(self.num_particles)
        return aw

    @typecheck
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ):
        # Kludge.
        log_weights = []
        for i in range(0, self.num_particles):
            key, sub_key = jax.random.split(key)
            (_, ps) = target.importance(sub_key)
            log_weights.append(ps.get_score())

        lws = tree_stack(log_weights)
        # END Kludge.

        tw = jax.scipy.special.logsumexp(lws)
        aw = tw - jnp.log(self.num_particles)
        return aw


@dataclass
class CustomSIR(SPAlgorithm):
    num_particles: Int
    proposal: SPDistribution
    proposal_args: Tuple

    def flatten(self):
        return (
            self.proposal,
            self.proposal_args,
        ), (self.num_particles,)

    @typecheck
    def estimate_recip_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: ChoiceMap,
        w: FloatArray,
    ):
        # BEGIN Kludge.
        particles = []
        weights = []
        for i in range(0, self.num_particles - 1):
            key, sub_key = jax.random.split(key)
            proposal_lws, ps = self.proposal.random_weighted(
                sub_key,
                target.constraints,
                *self.proposal_args,
            )
            particles.append(ps)
            weights.append(proposal_lws)

        weights.append(w)
        particles.append(latent_choices)

        proposal_lws = tree_stack(weights)
        ps = tree_stack(particles)
        # END Kludge.

        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        (_, particles) = jax.vmap(target.importance)(sub_keys, ps)
        lws = particles.get_score() - proposal_lws
        tw = jax.scipy.special.logsumexp(lws)
        aw = tw - jnp.log(self.num_particles)
        return aw

    @typecheck
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ):
        # BEGIN Kludge.
        particles = []
        weights = []
        for i in range(0, self.num_particles):
            key, sub_key = jax.random.split(key)
            proposal_lws, ps = self.proposal.random_weighted(
                sub_key, *self.proposal_args
            )
            particles.append(ps)
            weights.append(proposal_lws)

        proposal_lws = tree_stack(weights)
        ps = tree_stack(particles)
        # END Kludge.

        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        (_, particles) = jax.vmap(target.importance)(sub_keys, ps)
        lws = particles.get_score() - proposal_lws
        tw = jax.scipy.special.logsumexp(lws)
        aw = tw - jnp.log(self.num_particles)
        return aw


@dispatch
def sir(N: Int):
    return DefaultSIR(N)


@dispatch
def sir(
    N: Int,
    proposal: SPDistribution,
    proposal_args: Tuple,
):
    return CustomSIR(N, proposal, proposal_args)


@dataclass
class DefaultMarginal(SPDistribution):
    selection: Selection
    p: GenerativeFunction

    def flatten(self):
        return (self.select, self.p), ()

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Any:
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, args)
        weight = tr.get_score()
        choices = tr.strip()
        latent_choices = choices.filter(self.selection)
        other_choices = choices.filter(self.selection.complement())
        if isinstance(other_choices, EmptyChoiceMap):
            return (weight, ValueChoiceMap(latent_choices))
        else:
            q = sir(1, self.p, args)
            tgt = target(self.p, args, latent_choices)
            Z = q.estimate_recip_normalizing_constant(
                key,
                tgt,
                other_choices,
                weight,
            )
            return (Z, ValueChoiceMap(latent_choices))

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: ValueChoiceMap,
        *args,
    ) -> FloatArray:
        inner_choices = latent_choices.get_leaf_value()
        tgt = target(self.p, args, inner_choices)
        q = sir(1, self.p, args)
        Z = q.estimate_normalizing_constant(key, tgt)
        return Z


@dataclass
class CustomMarginal(SPDistribution):
    q: Callable[[Any, ...], SPAlgorithm]
    selection: Selection
    p: GenerativeFunction

    def flatten(self):
        return (self.selection, self.p), (self.q,)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Any:
        key, sub_key = jax.random.split(key)
        p_args, q_args = args
        tr = self.p.simulate(sub_key, p_args)
        weight = tr.get_score()
        choices = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        other_choices = choices.filter(self.selection.complement())
        tgt = target(self.p, p_args, latent_choices)
        alg = self.q(*q_args)
        Z = alg.estimate_recip_normalizing_constant(key, tgt, other_choices, weight)
        return (Z, ValueChoiceMap(latent_choices))

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: ValueChoiceMap,
        *args,
    ) -> FloatArray:
        inner_choices = latent_choices.get_leaf_value()
        (p_args, q_args) = args
        tgt = target(self.p, p_args, inner_choices)
        alg = self.q(*q_args)
        Z = alg.estimate_normalizing_constant(key, tgt)
        return Z


@dispatch
def marginal(
    selection: Selection,
    p: GenerativeFunction,
    q: Callable[[Any, ...], SPAlgorithm],
):
    return CustomMarginal.new(q, selection, p)


@dispatch
def marginal(
    selection: Selection,
    p: GenerativeFunction,
):
    return DefaultMarginal.new(selection, p)


##############
# Loss terms #
##############


@dispatch
def elbo(
    p: GenerativeFunction,
    q: SPDistribution,
    data: ChoiceMap,
):
    @adevjax.adev
    @typecheck
    def elbo_loss(p_args: Tuple, q_args: Tuple):
        tgt = target(p, p_args, data)
        variational_family = sir(1, q, q_args)
        key = adevjax.reap_key()
        w = variational_family.estimate_normalizing_constant(key, tgt)
        return w

    return adevjax.E(elbo_loss)


@dispatch
def elbo(
    p: GenerativeFunction,
    q: GenerativeFunction,
    data: ChoiceMap,
):
    marginal_q = marginal(AllSelection(), q)
    return elbo(
        p,
        marginal_q,
        data,
    )


@dispatch
def iwae_elbo(
    p: GenerativeFunction,
    q: SPDistribution,
    data: ChoiceMap,
    N: Int,
):
    @adevjax.adev
    @typecheck
    def elbo_loss(p_args: Tuple, q_args: Tuple):
        tgt = target(p, p_args, data)
        variational_family = sir(N, q, q_args)
        key = adevjax.reap_key()
        w = variational_family.estimate_normalizing_constant(key, tgt)
        return w

    return adevjax.E(elbo_loss)


@dispatch
def iwae_elbo(
    p: GenerativeFunction,
    q: GenerativeFunction,
    data: ChoiceMap,
    N: Int,
):
    marginal_q = marginal(AllSelection(), q)
    return iwae_elbo(
        p,
        marginal_q,
        data,
        N,
    )
