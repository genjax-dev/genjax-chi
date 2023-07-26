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

from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.gensp.gensp_distribution import (
    GenSPDistribution,
)
from genjax._src.generative_functions.distributions.gensp.target import Target
from genjax._src.generative_functions.distributions.gensp.utils import (
    static_check_supports,
)


def _logsumexp_with_extra(arr, x):
    max_arr = jnp.maximum(jnp.maximum(arr), x)
    return max_arr + jnp.log(jnp.sum(jnp.exp(arr - max_arr)) + jnp.exp(x - max - arr))


@dataclass
class Importance(GenSPDistribution):
    num_particles: int
    proposal: Union[None, GenSPDistribution]

    def flatten(self):
        return (), (self.num_particles, self.proposal)

    @typecheck
    @classmethod
    def new(cls, num_particles, proposal=None):
        return Importance(num_particles, proposal)

    def default_random_weighted(self, key, target: Target):
        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (lws, tr) = jax.vmap(target.p.importance, in_axes=(0, None, None))(
            sub_keys, target.constraints, target.args
        )
        tw = jax.scipy.special.logsumexp(lws)
        lnw = lws - tw
        aw = tw - np.log(self.num_particles)
        key, sub_key = jax.random.split(key)
        index = jax.random.categorical(sub_key, lnw)
        selected_particle = jtu.tree_map(lambda v: v[index], tr)
        return key, (
            selected_particle.get_score() - aw,
            target.get_latents(selected_particle),
        )

    def custom_random_weighted(self, key, target: Target):
        # Perform a compile-time trace type check.
        static_check_supports(target, self.proposal)

        key, *sub_keys = jax.random.split(key, self.num_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, particles = jax.vmap(self.proposal.simulate, in_axes=(0, None))(
            sub_keys, (target,)
        )
        constraints = target.constraints.safe_merge(particles.get_retval())
        _, (lws, _) = jax.vmap(target.p.importance, in_axes=(0, None, None))(
            sub_keys, constraints, target.args
        )
        lws = lws - particles.get_score()
        tw = jax.scipy.special.logsumexp(lws)
        lnw = lws - tw
        aw = tw - np.log(self.num_particles)
        key, sub_key = jax.random.split(key)
        index = jax.random.categorical(sub_key, lnw)
        selected_particle = jtu.tree_map(lambda v: v[index], particles)
        return key, (
            selected_particle.get_score() - aw,
            selected_particle.get_retval(),
        )

    def default_estimate_logpdf(self, key, chm, target):
        key, sub_keys = jax.random.split(key, self.num_particles)
        sub_keys = jnp.array(sub_keys)
        _, (lws, tr) = jax.vmap(target.p.importance, in_axes=(0, None, None))(
            sub_keys, target.constraints, target.args
        )
        merged = chm.safe_merge(target.constraints)
        key, retained_tr = target.p.importance(key, merged, target.args)
        constrained = target.constraints.get_selection()
        retained_w = retained_tr.project(constrained)
        lse = _logsumexp_with_extra(lws, retained_w)
        return key, retained_tr.get_score() - lse + np.log(self.num_particles)

    def custom_estimate_logpdf(self, key, chm, target):
        key, sub_keys = jax.random.split(key, self.num_particles)
        sub_keys = jnp.array(sub_keys)
        _, unchosen = jax.vmap(self.proposal.simulate, in_axes=(0, None))(
            sub_keys, (target,)
        )
        key, (retained_bwd, retained_tr) = self.proposal.importance(
            key, ValueChoiceMap.new(chm), (target,)
        )
        merged = target.constraints.safe_merge(unchosen.get_retval())
        key, sub_keys = jax.random.split(key, self.num_particles)
        sub_keys = jnp.array(sub_keys)
        _, (unchosen_fwd_lws, _) = jax.vmap(
            target.p.importance, in_axes=(0, None, None)
        )(sub_keys, merged, target.args)
        key, (retained_fwd, _) = target.p.importance(key, merged, target.args)
        unchosen_lws = unchosen_fwd_lws - unchosen.get_score()
        chosen_lw = retained_fwd - retained_bwd
        lse = _logsumexp_with_extra(unchosen_lws, chosen_lw)
        return (
            key,
            retained_tr.get_score() - lse + np.log(self.num_particles),
        )

    def random_weighted(self, key, target: Target):
        if self.proposal is None:
            return self.default_random_weighted(key, target)
        else:
            return self.custom_random_weighted(key, target)

    def estimate_logpdf(self, key, chm, target: Target):
        if self.proposal is None:
            return self.default_estimate_logpdf(key, chm, target)
        else:
            return self.custom_estimate_logpdf(key, chm, target)


##############
# Shorthands #
##############

importance = Importance.new
