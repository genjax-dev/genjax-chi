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
"""This module contains an implementation of (Symmetric divergence over datasets) from
Domke, 2021."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax._src.core.generative import (
    ChoiceMap,
    GenerativeFunction,
    Selection,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Int, PRNGKey, Tuple
from genjax._src.inference.core import Marginal, Target


class SymmetricDivergenceOverDatasets(Pytree):
    num_meta_p: Int = Pytree.static()
    num_meta_q: Int = Pytree.static()
    p: GenerativeFunction
    q: Marginal
    inf_selection: Selection

    def _estimate_log_ratio(self, key: PRNGKey, p_args: Tuple):
        # Inner functions -- to be mapped over.
        # Keys are folded in, for working memory.
        def _inner_p(key, index, choice, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = self.p.assess(new_key, choice, args)
            return w

        def _inner_q(key, index, choice, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = self.q.assess(new_key, choice, args)
            return w

        obs_target = self.inf_selection.complement()
        key_indices_p = jnp.arange(0, self.num_meta_p + 1)
        key_indices_q = jnp.arange(0, self.num_meta_q + 1)

        # (x, z) ~ p, log p(z, x) / q(z | x)
        key, tr = self.p.simulate(key, p_args)
        choice = tr.get_choices().strip()
        latent_choice = self.inf_selection.filter(choice)
        obs_choice = obs_target.filter(choice)
        (latent_choice, obs_choice) = (
            latent_choice.strip(),
            obs_choice.strip(),
        )

        # Compute estimate of log p(z, x)
        key, sub_key = jax.random.split(key)
        fwd_weights = jax.vmap(_inner_p, in_axes=(None, 0, None, None))(
            sub_key, key_indices_p, choice, p_args
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(self.num_meta_p)

        # Compute estimate of log q(z | x)
        constraints = obs_choice
        target = Target(self.p, p_args, constraints)
        latent_choice = ChoiceMap.v(latent_choice)
        key, sub_key = jax.random.split(key)
        bwd_weights = jax.vmap(_inner_q, in_axes=(None, 0, None, None))(
            sub_key, key_indices_q, latent_choice, (target,)
        )
        bwd_weight = logsumexp(bwd_weights) - jnp.log(self.num_meta_p)

        # log p(z, x) / q(z | x)
        logpq_fwd = fwd_weight - bwd_weight

        # z' ~ q, log p(z', x) / q(z', x)
        key, inftr = self.q.simulate(key, (target,))
        (inf_choice,) = inftr.get_retval()

        # Compute estimate of log p(z', x)
        key, sub_keys = jax.random.split(key)
        merged = obs_choice.safe_merge(inf_choice)
        sub_keys = jnp.array(sub_keys)
        fwd_weights = jax.vmap(_inner_p, in_axes=(None, 0, None, None))(
            sub_key, key_indices_p, merged, p_args
        )
        fwd_weight_p = logsumexp(fwd_weights) - jnp.log(self.num_meta_p)

        # Compute estimate of log q(z' | x)
        inf_choice = inftr.get_choices()
        key, sub_key = jax.random.split(key)
        bwd_weights = jax.vmap(_inner_q, in_axes=(None, 0, None, None))(
            sub_key, key_indices_q, inf_choice, (target,)
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
