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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import genjax.prox as prox
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Selection
from genjax.core.datatypes import ValueChoiceMap


def entropy_lower_bound(
    model: GenerativeFunction,
    proposal: prox.ProxDistribution,
    targets: Selection,
    M,
):
    def _inner(key, model_args):
        key, tr = model.simulate(key, model_args)
        obs_targets = targets.complement()
        observations, _ = obs_targets.filter(tr.get_choices().strip())
        target = prox.Target(model, None, model_args, observations)
        key, *sub_keys = jax.random.split(key, M + 1)
        sub_keys = jnp.array(sub_keys)
        _, tr_q = jax.vmap(proposal.simulate, in_axes=(0, None))(
            sub_keys, (target,)
        )
        (chm,) = tr_q.get_retval()
        chm_axes = jtu.tree_map(lambda v: 0, chm)
        observations_axes = jtu.tree_map(lambda v: None, observations)
        choices = observations.merge(chm)
        choices_axes = observations_axes.merge(chm_axes)
        key, *sub_keys = jax.random.split(key, M + 1)
        sub_keys = jnp.array(sub_keys)
        _, (log_p, _) = jax.vmap(
            model.importance, in_axes=(0, choices_axes, None)
        )(sub_keys, choices, model_args)
        log_q = tr_q.get_score()
        log_w = log_p - log_q
        return key, jnp.mean(log_w), (log_p, log_q)

    return _inner


def entropy_upper_bound(
    model: GenerativeFunction,
    proposal: prox.ProxDistribution,
    targets: Selection,
    M,
):
    def _inner(key, model_args):
        key, tr = model.simulate(key, model_args)
        log_p = tr.get_score()
        chm = tr.get_choices().strip()
        latents, _ = targets.filter(chm)
        observations, _ = targets.complement().filter(chm)
        target = prox.Target(model, None, model_args, observations)
        key, *sub_keys = jax.random.split(key, M + 1)
        sub_keys = jnp.array(sub_keys)
        _, (log_q, tr_q) = jax.vmap(
            proposal.importance, in_axes=(0, None, None)
        )(sub_keys, ValueChoiceMap(latents), (target,))
        log_w = log_q - log_p
        return key, -jnp.mean(log_w), (log_p, log_q)

    return _inner


def entropy_estimators_via_inference(
    model: GenerativeFunction,
    proposal: prox.ProxDistribution,
    targets: Selection,
    N,
    M,
):
    lower_bound_func = entropy_lower_bound(model, proposal, targets, M)
    upper_bound_func = entropy_upper_bound(model, proposal, targets, M)

    def _inner(key, model_args):
        key, *sub_keys = jax.random.split(key, N + 1)
        sub_keys = jnp.array(sub_keys)
        _, lower_bound, (llog_p, llog_q) = jax.vmap(
            lower_bound_func, in_axes=(0, None)
        )(sub_keys, model_args)
        key, *sub_keys = jax.random.split(key, N + 1)
        sub_keys = jnp.array(sub_keys)
        _, upper_bound, (ulog_p, ulog_q) = jax.vmap(
            upper_bound_func, in_axes=(0, None)
        )(sub_keys, model_args)
        d = {"lower": (llog_p, llog_q), "upper": (ulog_p, ulog_q)}
        return key, (-jnp.mean(upper_bound), -jnp.mean(lower_bound)), d

    return _inner


eevi = entropy_estimators_via_inference
