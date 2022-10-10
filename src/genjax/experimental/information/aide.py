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
An implementation of (Auxiliary inference divergence estimator)
from Cusumano-Towner et al, 2017.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax.core.datatypes import GenerativeFunction


def estimate_log_ratio(
    p: GenerativeFunction, q: GenerativeFunction, mp: int, mq: int
):
    def _inner(key, p_args: Tuple, q_args: Tuple):
        key, tr = p.simulate(key, p_args)
        chm = tr.get_choices().strip_metadata()
        key, *sub_keys = jax.random.split(key, mp + 1)
        sub_keys = jnp.array(sub_keys)
        _, (fwd_weights, _) = jax.vmap(p.importance, in_axes=(0, None, None))(
            sub_keys,
            chm,
            p_args,
        )
        key, *sub_keys = jax.random.split(key, mq + 1)
        sub_keys = jnp.array(sub_keys)
        _, (bwd_weights, _) = jax.vmap(q.importance, in_axes=(0, None, None))(
            sub_keys,
            chm,
            q_args,
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(mp)
        bwd_weight = logsumexp(bwd_weights) - jnp.log(mq)
        return key, fwd_weight - bwd_weight

    return _inner


def auxiliary_inference_divergence_estimator(
    p: GenerativeFunction, q: GenerativeFunction, mp: int, mq: int
):
    def _inner(key, p_args, q_args):
        key, logpq = estimate_log_ratio(p, q, mp, mq)(key, p_args, q_args)
        key, logqp = estimate_log_ratio(q, p, mq, mp)(key, q_args, p_args)
        return key, logpq + logqp, (logpq, logqp)

    return _inner


aide = auxiliary_inference_divergence_estimator
