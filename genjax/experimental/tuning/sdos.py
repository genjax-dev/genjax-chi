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

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from genjax.core.datatypes import GenerativeFunction, Selection
from typing import Tuple


def estimate_log_ratio(
    p: GenerativeFunction,
    q: GenerativeFunction,
    inf_target: Selection,
    mp: int,
    mq: int,
):
    def _inner(key, p_args: Tuple, q_args: Tuple):
        obs_target = inf_target.complement()

        # (x, z) ~ p, log p(z, x) / q(z | x)
        key, tr = p.simulate(key, p_args)
        chm = tr.get_choices()
        latent_chm = inf_target.filter(chm)
        obs_chm = obs_target.filter(chm)

        # Compute estimate of log p(z, x)
        key, *subkeys = jax.random.split(key, mp + 1)
        subkeys = jnp.array(subkeys)
        _, (fwd_weights, _) = jax.vmap(p.importance, in_axes=(0, None, None))(
            subkeys,
            chm,
            p_args,
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(mp)

        # Compute estimate of log q(z | x)
        key, *subkeys = jax.random.split(key, mq + 1)
        subkeys = jnp.array(subkeys)
        _, (bwd_weights, _) = jax.vmap(q.importance, in_axes=(0, None, None))(
            subkeys,
            latent_chm,
            (chm, *q_args),
        )
        bwd_weight = logsumexp(bwd_weights) - jnp.log(mq)

        # log p(z, x) / q(z | x)
        logpq = fwd_weight - bwd_weight

        # z' ~ q, log p(z', x) / q(z', x)
        key, inftr = q.simulate(key, (obs_chm, *q_args))
        inf_chm = inftr.get_choices()

        # Compute estimate of log p(z', x)
        key, *subkeys = jax.random.split(key, mp + 1)
        subkeys = jnp.array(subkeys)
        _, (fwd_weights, _) = jax.vmap(p.importance, in_axes=(0, None, None))(
            subkeys,
            obs_chm.merge(inf_chm),
            p_args,
        )
        fwd_weight_p = logsumexp(fwd_weights) - jnp.log(mp)

        # Compute estimate of log q(z' | x)
        key, *subkeys = jax.random.split(key, mp + 1)
        subkeys = jnp.array(subkeys)
        _, (fwd_weights, _) = jax.vmap(q.importance, in_axes=(0, None, None))(
            subkeys,
            inf_chm,
            (obs_chm, *q_args),
        )
        bwd_weight_p = logsumexp(bwd_weights) - jnp.log(mq)

        # log p(z', x) / q(z'| x)
        logpq_p = fwd_weight_p - bwd_weight_p

        return key, logpq - logpq_p

    return _inner


def sdos(
    p: GenerativeFunction,
    q: GenerativeFunction,
    inf_target: Selection,
    mp: int,
    mq: int,
):
    def _inner(key, p_args: Tuple, q_args: Tuple):
        key, logpq = estimate_log_ratio(p, q, inf_target, mp, mq)(
            key, p_args, q_args
        )
        key, logqp = estimate_log_ratio(q, p, inf_target, mq, mp)(
            key, q_args, p_args
        )
        return key, logpq + logqp

    return _inner
