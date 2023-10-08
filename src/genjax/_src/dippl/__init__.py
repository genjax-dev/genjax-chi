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

from dataclasses import dataclass

import adevjax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from adevjax import ADEVPrimitive
from adevjax import E
from adevjax import add_cost
from adevjax import adev
from adevjax import flip_enum
from adevjax import geometric_reinforce
from adevjax import grab_key
from adevjax import mv_normal_diag_reparam
from adevjax import mv_normal_reparam
from adevjax import normal_reinforce
from adevjax import normal_reparam
from adevjax import sample_with_key

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
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
class DefaultImportanceEnum(ChoiceMapDistribution):
    num_particles: Int

    def flatten(self):
        return (), (self.num_particles,)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        target: Target,
    ):
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        (lws, tr) = jax.vmap(target.importance, in_axes=(0, None))(
            sub_keys, EmptyChoiceMap()
        )
        tw = jax.scipy.special.logsumexp(lws)
        lnw = lws - tw
        aw = tw - np.log(self.num_particles)
        idx = sample_with_key(
            adevjax.categorical_enum, key, lnw
        )  # enumerate over all possible index choices
        selected_particle = jtu.tree_map(lambda v: v[idx], tr)
        return (
            selected_particle.get_score() - aw,
            ValueChoiceMap(target.get_latents(selected_particle)),
        )

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        chm: ValueChoiceMap,
        target: Target,
    ):
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(key, self.num_particles - 1)
        (lws, _) = jax.vmap(target.importance, in_axes=(0, None))(
            sub_keys, EmptyChoiceMap()
        )
        inner_chm = chm.get_leaf_value()
        assert isinstance(inner_chm, ChoiceMap)
        (retained_w, retained_tr) = target.importance(key, inner_chm)
        lse = _logsumexp_with_extra(lws, retained_w)
        return retained_tr.get_score() - lse + np.log(self.num_particles)


@dataclass
class CustomImportanceEnum(ChoiceMapDistribution):
    num_particles: Int
    proposal: ChoiceMapDistribution

    def flatten(self):
        return (self.proposal,), (self.num_particles,)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        target: Target,
    ):
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        (ws, ps) = jax.vmap(self.proposal.random_weighted, in_axes=(0, None))(
            sub_keys, target
        )
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        (lws, _) = jax.vmap(target.importance, in_axes=(0, None))(sub_keys, ps)
        lws = lws - particles.get_score()
        tw = jax.scipy.special.logsumexp(lws)
        lnw = lws - tw
        aw = tw - np.log(self.num_particles)
        idx = sample_with_key(
            adevjax.categorical_enum, key, lnw
        )  # enumerate over all possible index choices
        selected_particle = jtu.tree_map(lambda v: v[idx], particles)
        return (selected_particle.get_score() - aw, selected_particle)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        chm: ValueChoiceMap,
        target: Target,
    ):
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        unchosen_bwd_lws, unchosen = jax.vmap(
            self.proposal.random_weighted, in_axes=(0, None)
        )(sub_keys, target)
        key, sub_key = jax.random.split(key)
        retained_bwd = self.proposal.estimate_logpdf(sub_key, chm, target)
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles - 1)
        (unchosen_fwd_lws, _) = jax.vmap(target.importance, in_axes=(0, 0))(
            sub_keys, unchosen.get_retval()
        )
        inner_chm = chm.get_leaf_value()
        assert isinstance(inner_chm, ChoiceMap)
        (retained_fwd, retained_tr) = target.importance(key, inner_chm)
        unchosen_lws = unchosen_fwd_lws - unchosen_bwd_lws
        chosen_lw = retained_fwd - retained_bwd
        lse = _logsumexp_with_extra(unchosen_lws, chosen_lw)
        return retained_tr.get_score() - lse + np.log(self.num_particles)


@dispatch
def importance_enum(num_particles: Int):
    return DefaultImportanceEnum.new(num_particles)


##################################
# Differentiable loss primitives #
##################################


def upper(prim: Distribution):
    def _inner(*args):
        key = grab_key()
        (w, v) = prim.random_weighted(key, *args)
        add_cost(-w)
        return v

    return _inner


def lower(prim: Distribution):
    def _inner(v, *args):
        key = grab_key()
        w = prim.estimate_logpdf(key, v, *args)
        add_cost(w)

    return _inner


def loss(fn: Callable):
    @adev
    def _inner(*args):
        v = fn(*args)
        if v is None:
            return 0.0
        else:
            return v

    return E(_inner)
