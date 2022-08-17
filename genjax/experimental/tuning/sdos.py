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

from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from genjax.generative_function import simulate, importance


def estimate_log_ratio(p: Callable, q: Callable, inf_target, mp: int, mq: int):
    def __inner(key, p_args, q_args):
        obs_target = inf_target.complement()

        # (x, z) ~ p, log p(z, x) / q(z | x)
        key, tr = simulate(p)(key, p_args)
        chm = tr.get_choices()
        latent_chm = inf_target.filter(chm)
        obs_chm = obs_target.filter(chm)

        # Compute estimate of log p(z, x)
        key, *subkeys = jax.random.split(key, mp + 1)
        subkeys = jnp.array(subkeys)
        _, (fwd_weights, _) = jax.vmap(importance(p), in_axes=(0, None, None))(
            subkeys, chm, p_args
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(mp)

        # Compute estimate of log q(z | x)
        key, *subkeys = jax.random.split(key, mq + 1)
        subkeys = jnp.array(subkeys)
        _, (bwd_weights, _) = jax.vmap(importance(q), in_axes=(0, None, None))(
            subkeys, latent_chm, (chm, *q_args)
        )
        bwd_weight = logsumexp(bwd_weights) - jnp.log(mq)

        # log p(z, x) / q(z | x)
        logpq = fwd_weight - bwd_weight

        # z' ~ q, log p(z', x) / q(z', x)
        key, inftr = simulate(q)(key, (obs_chm, *q_args))
        inf_chm = inftr.get_choices()

        # Compute estimate of log p(z', x)
        key, *subkeys = jax.random.split(key, mp + 1)
        subkeys = jnp.array(subkeys)
        _, (fwd_weights, _) = jax.vmap(importance(p), in_axes=(0, None, None))(
            subkeys, obs_chm.merge(inf_chm), p_args
        )
        fwd_weight_p = logsumexp(fwd_weights) - jnp.log(mp)

        # Compute estimate of log q(z' | x)
        key, *subkeys = jax.random.split(key, mp + 1)
        subkeys = jnp.array(subkeys)
        _, (fwd_weights, _) = jax.vmap(importance(q), in_axes=(0, None, None))(
            subkeys, inf_chm, (obs_chm, *q_args)
        )
        bwd_weight_p = logsumexp(bwd_weights) - jnp.log(mq)

        # log p(z', x) / q(z'| x)
        logpq_p = fwd_weight_p - bwd_weight_p

        return key, logpq - logpq_p

    return lambda key, p_args, q_args: __inner(key, p_args, q_args)


def sdos(p: Callable, q: Callable, inf_target, mp: int, mq: int):
    def __inner(key, p_args, q_args):
        key, logpq = estimate_log_ratio(p, q, inf_target, mp, mq)(
            key, p_args, q_args
        )
        key, logqp = estimate_log_ratio(q, p, inf_target, mq, mp)(
            key, q_args, p_args
        )
        return key, logpq + logqp

    return lambda key, p_args, q_args: __inner(key, p_args, q_args)
