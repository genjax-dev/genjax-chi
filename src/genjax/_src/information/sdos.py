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

"""This module contains an implementation of (Symmetric divergence over
datasets) from Domke, 2021."""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import ValueChoiceMap
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.prox import ProxDistribution
from genjax._src.prox import Target


@dataclasses.dataclass
class SymmetricDivergenceOverDatasets(Pytree):
    p: GenerativeFunction
    q: ProxDistribution
    inf_selection: Selection
    num_meta_p: Int
    num_meta_q: Int

    def flatten(self):
        return (), (
            self.p,
            self.q,
            self.inf_selection,
            self.num_meta_p,
            self.num_meta_q,
        )

    def _estimate_log_ratio(self, key: PRNGKey, p_args: Tuple):
        # Inner functions -- to be mapped over.
        # Keys are folded in, for working memory.
        def _inner_p(key, index, chm, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = self.p.importance(new_key, chm, args)
            return w

        def _inner_q(key, index, chm, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = self.q.importance(new_key, chm, args)
            return w

        obs_target = self.inf_selection.complement()
        key_indices_p = jnp.arange(0, self.num_meta_p + 1)
        key_indices_q = jnp.arange(0, self.num_meta_q + 1)

        # (x, z) ~ p, log p(z, x) / q(z | x)
        key, tr = self.p.simulate(key, p_args)
        chm = tr.get_choices().strip()
        latent_chm = self.inf_selection.filter(chm)
        obs_chm = obs_target.filter(chm)
        (latent_chm, obs_chm) = (
            latent_chm.strip(),
            obs_chm.strip(),
        )

        # Compute estimate of log p(z, x)
        key, sub_key = jax.random.split(key)
        fwd_weights = jax.vmap(_inner_p, in_axes=(None, 0, None, None))(
            sub_key, key_indices_p, chm, p_args
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(self.num_meta_p)

        # Compute estimate of log q(z | x)
        constraints = obs_chm
        target = Target(self.p, p_args, constraints)
        latent_chm = ValueChoiceMap.new(latent_chm)
        key, sub_key = jax.random.split(key)
        bwd_weights = jax.vmap(_inner_q, in_axes=(None, 0, None, None))(
            sub_key, key_indices_q, latent_chm, (target,)
        )
        bwd_weight = logsumexp(bwd_weights) - jnp.log(self.num_meta_p)

        # log p(z, x) / q(z | x)
        logpq_fwd = fwd_weight - bwd_weight

        # z' ~ q, log p(z', x) / q(z', x)
        key, inftr = self.q.simulate(key, (target,))
        (inf_chm,) = inftr.get_retval()

        # Compute estimate of log p(z', x)
        key, sub_keys = jax.random.split(key)
        merged = obs_chm.merge(inf_chm)
        sub_keys = jnp.array(sub_keys)
        fwd_weights = jax.vmap(_inner_p, in_axes=(None, 0, None, None))(
            sub_key, key_indices_p, merged, p_args
        )
        fwd_weight_p = logsumexp(fwd_weights) - jnp.log(self.num_meta_p)

        # Compute estimate of log q(z' | x)
        inf_chm = inftr.get_choices()
        key, sub_key = jax.random.split(key)
        bwd_weights = jax.vmap(_inner_q, in_axes=(None, 0, None, None))(
            sub_key, key_indices_q, inf_chm, (target,)
        )
        bwd_weight_p = logsumexp(bwd_weights) - jnp.log(self.num_meta_q)

        # log p(z', x) / q(z'| x)
        logpq_bwd = fwd_weight_p - bwd_weight_p

        return key, logpq_fwd - logpq_bwd, (logpq_fwd, logpq_bwd)

    def estimate(self, key: PRNGKey, p_args: Tuple):
        key, est, (fwd, bwd) = self._estimate_log_ratio(key, p_args)
        return key, est, (fwd, bwd)

    def __call__(self, key: PRNGKey, p_args: Tuple):
        return self.estimate(key, p_args)


##############
# Shorthands #
##############

sdos = SymmetricDivergenceOverDatasets.new
