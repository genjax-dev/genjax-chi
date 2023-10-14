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
from adevjax import add_cost
from adevjax import flip_enum
from adevjax import geometric_reinforce
from adevjax import grab_key
from adevjax import mv_normal_diag_reparam
from adevjax import mv_normal_reparam
from adevjax import normal_reinforce
from adevjax import normal_reparam
from adevjax import sample_with_key

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.utilities import tree_stack
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.generative_functions.distributions.distribution import ExactDensity
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_bernoulli,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    tfp_categorical,
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
from genjax._src.gensp.choice_map_distribution import ChoiceMapDistribution
from genjax._src.gensp.sp_distribution import SPDistribution
from genjax._src.gensp.target import Target


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
    lambda v, p: tfp_bernoulli.logpdf(v, probs=p),
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
        tr: Trace,
        w: FloatArray,
    ):
        pass


@dataclass
class IWAEImportance(SPAlgorithm):
    num_particles: Int
    proposal: ChoiceMapDistribution
    proposal_args: Tuple

    def flatten(self):
        return (self.proposal, self.proposal_args), (self.num_particles,)

    @typecheck
    def estimate_recip_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        tr: Trace,
        w: FloatArray,
    ):
        pass

    @typecheck
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ):
        # Kludge.
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


def iwae_importance(
    N: Int,
    proposal: ChoiceMapDistribution,
    proposal_args: Tuple,
):
    return IWAEImportance(N, proposal, proposal_args)


@dataclass
class Marginal(Distribution):
    addr: Any
    p: GenerativeFunction
    q: SPAlgorithm

    def flatten(self):
        return (self.p, self.q), (self.addr,)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Any:
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, args)
        weight = tr.get_score()
        choices = tr.get_choices()
        val = choices[self.addr]
        selection = select(self.addr).complement()
        other_choices = choices.filter(selection)
        tgt = target(self.p, args, choice_map({self.addr: val}))
        Z = self.q.estimate_recip_normalizing_constant(
            key, ValueChoiceMap.new(other_choices), tgt
        )
        weight -= Z
        return (weight, val)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        val: Any,
        *args,
    ) -> FloatArray:
        pass


##################################
# Differentiable loss primitives #
##################################


def upper(prim: Distribution):
    def _inner(*args):
        key = grab_key()
        return prim.random_weighted(key, *args)

    return _inner


def do_upper(prim: Distribution):
    def _inner(*args):
        key = grab_key()
        (w, v) = prim.random_weighted(key, *args)
        add_cost(-w)
        return v

    return _inner


def lower(prim: Distribution):
    def _inner(v, *args):
        key = grab_key()
        return prim.estimate_logpdf(key, v, *args)

    return _inner


def do_lower(prim: Distribution):
    def _inner(v, *args):
        key = grab_key()
        w = prim.estimate_logpdf(key, v, *args)
        add_cost(w)

    return _inner


#####
# Losses
#####


def elbo(
    p: GenerativeFunction,
    nondiff_p_args: Tuple,
    q: SPDistribution,
    nondiff_q_args: Tuple,
    data: ChoiceMap,
):
    @adevjax.adev
    def elbo_loss(diff_p_args, diff_q_args):
        tgt = gensp.target(p, (*diff_p_args, *nondiff_p_args), data)
        key = adevjax.grab_key()
        variational_family = iwae_importance(1, q, (diff_q_args, nondiff_q_args))
        w = variational_family.estimate_normalizing_constant(key, tgt)
        return w

    return adevjax.E(elbo_loss)


def iwae_elbo(
    p: GenerativeFunction,
    nondiff_p_args: Tuple,
    q: SPDistribution,
    nondiff_q_args: Tuple,
    data: ChoiceMap,
    N: Int,
):
    @adevjax.adev
    def elbo_loss(diff_p_args, diff_q_args):
        tgt = gensp.target(p, (*diff_p_args, *nondiff_p_args), data)
        key = adevjax.grab_key()
        variational_family = iwae_importance(N, q, (diff_q_args, nondiff_q_args))
        w = variational_family.estimate_normalizing_constant(key, tgt)
        return w

    return adevjax.E(elbo_loss)
