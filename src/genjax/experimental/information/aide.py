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
This module contains an implementation of (Auxiliary inference divergence estimator) from Cusumano-Towner et al, 2017.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax.core.datatypes import GenerativeFunction


def estimate_log_ratio(
    p: GenerativeFunction,
    q: GenerativeFunction,
    mp: int,
    mq: int,
):
    def _inner(key, p_args: Tuple, q_args: Tuple):

        # Inner functions -- to be mapped over.
        # Keys are folded in, for working memory.
        def _inner_p(key, index, chm, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = p.importance(new_key, chm, args)
            return w

        def _inner_q(key, index, chm, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = q.importance(new_key, chm, args)
            return w

        key_indices_p = jnp.arange(0, mp + 1)
        key_indices_q = jnp.arange(0, mq + 1)

        key, tr = p.simulate(key, p_args)
        chm = tr.get_choices().strip()
        key, sub_key = jax.random.split(key)
        fwd_weights = jax.vmap(_inner_p, in_axes=(None, 0, None, None))(
            sub_key, key_indices_p, chm, p_args
        )
        key, sub_key = jax.random.split(key)
        bwd_weights = jax.vmap(_inner_q, in_axes=(None, 0, None, None))(
            sub_key, key_indices_q, chm, q_args
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(mp)
        bwd_weight = logsumexp(bwd_weights) - jnp.log(mq)
        return key, fwd_weight - bwd_weight

    return _inner


def auxiliary_inference_divergence_estimator(
    p: GenerativeFunction,
    q: GenerativeFunction,
    mp: int,
    mq: int,
):
    def _inner(key, p_args, q_args):
        key, logpq = estimate_log_ratio(p, q, mp, mq)(key, p_args, q_args)
        key, logqp = estimate_log_ratio(q, p, mq, mp)(key, q_args, p_args)
        return key, logpq + logqp, (logpq, logqp)

    return _inner


aide = auxiliary_inference_divergence_estimator
