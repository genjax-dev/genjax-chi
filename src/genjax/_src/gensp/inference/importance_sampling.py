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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.gensp.choice_map_distribution import ChoiceMapDistribution
from genjax._src.gensp.target import Target


def _logsumexp_with_extra(arr, x):
    max_arr = jnp.maximum(jnp.maximum(arr), x)
    return max_arr + jnp.log(jnp.sum(jnp.exp(arr - max_arr)) + jnp.exp(x - max - arr))


@dataclass
class DefaultImportance(ChoiceMapDistribution):
    num_particles: int

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
        index = jax.random.categorical(key, lnw)
        selected_particle = jtu.tree_map(lambda v: v[index], tr)
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
class CustomImportance(ChoiceMapDistribution):
    num_particles: int
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
        particles = jax.vmap(self.proposal.simulate, in_axes=(0, None))(
            sub_keys, (target,)
        )
        constraints = target.constraints.safe_merge(particles.get_retval())
        key, sub_key = jax.random.split(key)
        sub_keys = jax.random.split(sub_key, self.num_particles)
        (lws, _) = jax.vmap(target.importance, in_axes=(0, None))(sub_keys, constraints)
        lws = lws - particles.get_score()
        tw = jax.scipy.special.logsumexp(lws)
        lnw = lws - tw
        aw = tw - np.log(self.num_particles)
        index = jax.random.categorical(sub_key, lnw)
        selected_particle = jtu.tree_map(lambda v: v[index], particles)
        return (
            selected_particle.get_score() - aw,
            selected_particle.get_retval(),
        )

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


##############
# Shorthands #
##############


@dispatch
def importance_sampler(N: Int):
    return DefaultImportance.new(N)


@dispatch
def importance_sampler(N: Int, proposal: ChoiceMapDistribution):
    return CustomImportance.new(N, proposal)
