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

"""
An implementation of (Symmetric divergence over datasets)
from Domke, 2021.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import genjax.experimental.prox as prox
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Selection
from genjax.core.datatypes import ValueChoiceMap


def estimate_log_ratio(
    p: GenerativeFunction,
    q: prox.ProxDistribution,
    inf_selection: Selection,
    mp: int,
    mq: int,
):
    def _inner(key, p_args: Tuple):
        obs_target = inf_selection.complement()

        # (x, z) ~ p, log p(z, x) / q(z | x)
        key, tr = p.simulate(key, p_args)
        chm = tr.get_choices().strip_metadata()
        latent_chm, _ = inf_selection.filter(chm)
        obs_chm, _ = obs_target.filter(chm)
        (latent_chm, obs_chm) = (
            latent_chm.strip_metadata(),
            obs_chm.strip_metadata(),
        )

        # Compute estimate of log p(z, x)
        key, *sub_keys = jax.random.split(key, mp + 1)
        sub_keys = jnp.array(sub_keys)
        _, (fwd_weights, _) = jax.vmap(p.importance, in_axes=(0, None, None))(
            sub_keys,
            chm,
            p_args,
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(mp)

        # Compute estimate of log q(z | x)
        constraints = obs_chm
        target = prox.Target(p, None, p_args, constraints)
        latent_chm = ValueChoiceMap.new(latent_chm)
        key, *sub_keys = jax.random.split(key, mq + 1)
        sub_keys = jnp.array(sub_keys)
        _, (bwd_weights, _) = jax.vmap(q.importance, in_axes=(0, None, None))(
            sub_keys,
            latent_chm,
            (target,),
        )
        bwd_weight = logsumexp(bwd_weights) - jnp.log(mq)

        # log p(z, x) / q(z | x)
        logpq_fwd = fwd_weight - bwd_weight

        # z' ~ q, log p(z', x) / q(z', x)
        key, inftr = q.simulate(key, (target,))
        (inf_chm,) = inftr.get_retval()

        # Compute estimate of log p(z', x)
        key, *sub_keys = jax.random.split(key, mp + 1)
        merged = obs_chm.merge(inf_chm)
        sub_keys = jnp.array(sub_keys)
        _, (fwd_weights, _) = jax.vmap(p.importance, in_axes=(0, None, None))(
            sub_keys,
            merged,
            p_args,
        )
        fwd_weight_p = logsumexp(fwd_weights) - jnp.log(mp)

        # Compute estimate of log q(z' | x)
        inf_chm = inftr.get_choices()
        key, *sub_keys = jax.random.split(key, mq + 1)
        sub_keys = jnp.array(sub_keys)
        _, (bwd_weights, _) = jax.vmap(q.importance, in_axes=(0, None, None))(
            sub_keys,
            inf_chm,
            (target,),
        )
        bwd_weight_p = logsumexp(bwd_weights) - jnp.log(mq)

        # log p(z', x) / q(z'| x)
        logpq_bwd = fwd_weight_p - bwd_weight_p

        return key, logpq_fwd - logpq_bwd, (logpq_fwd, logpq_bwd)

    return _inner


def symmetric_divergence_over_datasets(
    p: GenerativeFunction,
    q: prox.ProxDistribution,
    inf_selection: Selection,
    mp: int,
    mq: int,
):
    def _inner(key, p_args: Tuple):
        key, est, (fwd, bwd) = estimate_log_ratio(p, q, inf_selection, mp, mq)(
            key, p_args
        )
        return key, est, (fwd, bwd)

    return _inner


sdos = symmetric_divergence_over_datasets
