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
from genjax.core.datatypes import GenerativeFunction, Selection
import genjax.experimental.prox as prox


def entropy_lower_bound(
    model: GenerativeFunction,
    proposal: prox.ProxDistribution,
    targets: Selection,
    M,
):
    def _inner(key, model_args):
        key, tr = model.simulate(key, model_args)
        observations, _ = targets.filter(tr.get_choices().strip_metadata())
        target = prox.Target(model, None, model_args, observations)
        key, *sub_keys = jax.random.split(key, M + 1)
        sub_keys = jnp.array(sub_keys)
        _, tr_q = jax.vmap(proposal.simulate, in_axes=(0, None))(
            sub_keys, (target,)
        )
        chm = tr_q.get_retval()
        chm_axes = jtu.tree_map(lambda v: 0, chm)
        observations_axes = jtu.tree_map(lambda v: None, observations)
        choices = observations.merge(chm)
        choices_axes = observations_axes.merge(chm_axes)
        key, *sub_keys = jax.random.split(key, M + 1)
        sub_keys = jnp.array(sub_keys)
        _, (w, _) = jax.vmap(model.importance, in_axes=(0, choices_axes, None))(
            sub_keys, choices, model_args
        )
        log_w = w - tr_q.get_score()
        return key, jnp.mean(log_w)

    return _inner


def entropy_upper_bound(
    model: GenerativeFunction,
    proposal: prox.ProxDistribution,
    targets: Selection,
    M,
):
    def _inner(key, model_args):
        key, tr = model.simulate(key, model_args)
        score = tr.get_score()
        chm = tr.get_choices().strip_metadata()
        observations, _ = targets.filter(chm)
        latents, _ = targets.complement().filter(chm)
        target = prox.Target(model, None, model_args, observations)
        key, *sub_keys = jax.random.split(key, M + 1)
        sub_keys = jnp.array(sub_keys)
        _, (log_q, tr_q) = jax.vmap(proposal.importance, in_axes=(0, None))(
            sub_keys, latents, (target,)
        )
        log_w = log_q - score
        return key, -jnp.mean(log_w)

    return _inner


def interval_entropy_estimator(
    model: GenerativeFunction,
    proposal: prox.ProxDistribution,
    targets: Selection,
    M,
):
    lower_bound_func = entropy_lower_bound(model, proposal, targets, M)
    upper_bound_func = entropy_upper_bound(model, proposal, targets, M)

    def _inner(key, model_args):
        key, lower_bound = lower_bound_func(key, model_args)
        key, upper_bound = upper_bound_func(key, model_args)
        return key, (lower_bound, upper_bound)

    return _inner


iee = interval_entropy_estimator
