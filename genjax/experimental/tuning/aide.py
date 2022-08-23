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

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from genjax.interface import simulate, importance
from genjax.core.datatypes import GenerativeFunction


def estimate_log_ratio(
    p: GenerativeFunction, q: GenerativeFunction, mp: int, mq: int
):
    def __inner(key, p_args, q_args):
        key, tr = simulate(p)(key, p_args)
        chm = tr.get_choices()
        key, *subkeys = jax.random.split(key, mp + 1)
        subkeys = jnp.array(subkeys)
        _, (fwd_weights, _) = jax.vmap(importance(p), in_axes=(0, None, None))(
            subkeys, chm, p_args
        )
        key, *subkeys = jax.random.split(key, mq + 1)
        subkeys = jnp.array(subkeys)
        _, (bwd_weights, _) = jax.vmap(importance(q), in_axes=(0, None, None))(
            subkeys, chm, q_args
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(mp)
        bwd_weight = logsumexp(bwd_weights) - jnp.log(mq)
        return key, fwd_weight - bwd_weight

    return lambda key, p_args, q_args: __inner(key, p_args, q_args)


def aide(p: GenerativeFunction, q: GenerativeFunction, mp: int, mq: int):
    def __inner(key, p_args, q_args):
        key, logpq = estimate_log_ratio(p, q, mp, mq)(key, p_args, q_args)
        key, logqp = estimate_log_ratio(q, p, mq, mp)(key, q_args, p_args)
        return key, logpq + logqp

    return lambda key, p_args, q_args: __inner(key, p_args, q_args)
