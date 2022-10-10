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

from typing import Tuple

import jax
import jax.numpy as jnp

from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import GenerativeFunction


def bootstrap_importance_sampling(model: GenerativeFunction, n_particles: int):
    def _inner(key, model_args: Tuple, observations: ChoiceMap):
        key, *subkeys = jax.random.split(key, n_particles + 1)
        subkeys = jnp.array(subkeys)
        _, (lws, trs) = jax.vmap(model.importance, in_axes=(0, None, None))(
            subkeys,
            observations,
            model_args,
        )
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(n_particles)
        return key, (trs, log_normalized_weights, log_ml_estimate)

    return _inner


def proposal_importance_sampling(
    model: GenerativeFunction, proposal: GenerativeFunction, n_particles: int
):
    def _inner(
        key,
        model_args: Tuple,
        proposal_args: Tuple,
        observations: ChoiceMap,
    ):
        key, *subkeys = jax.random.split(key, n_particles + 1)
        subkeys = jnp.array(subkeys)
        _, p_trs = jax.vmap(proposal.simulate, in_axes=(0, None, None))(
            subkeys,
            observations,
            model_args,
        )
        observations = jax.tree_util.map(
            lambda v: jnp.repeats(v, n_particles), observations
        )
        chm = p_trs.get_choices().merge(observations)
        key, *subkeys = jax.random.split(key, n_particles + 1)
        subkeys = jnp.array(subkeys)
        _, (lws, m_trs) = jax.vmap(model.importance, in_axes=(0, 0, None))(
            subkeys,
            chm,
            model_args,
        )
        lws = lws - p_trs.get_score()
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(n_particles)
        return key, (m_trs, log_normalized_weights, log_ml_estimate)

    return _inner


def bootstrap_importance_resampling(
    model: GenerativeFunction, n_particles: int
):
    def _inner(key, obs: ChoiceMap, model_args: Tuple):
        key, *subkeys = jax.random.split(key, n_particles + 1)
        subkeys = jnp.array(subkeys)
        _, (lws, trs) = jax.vmap(model.importance, in_axes=(0, None, None))(
            subkeys, obs, model_args
        )
        log_total_weight = jax.scipy.special.logsumexp(lws)
        log_normalized_weights = lws - log_total_weight
        log_ml_estimate = log_total_weight - jnp.log(n_particles)
        ind = jax.random.categorical(key, log_normalized_weights)
        tr = jax.tree_util.tree_map(lambda v: v[ind], trs)
        lnw = log_normalized_weights[ind]
        return key, (tr, lnw, log_ml_estimate)

    return _inner
