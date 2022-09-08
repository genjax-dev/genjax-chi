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
from genjax.core.datatypes import ChoiceMap, GenerativeFunction, IndexMask
from genjax.core.pytree import tree_stack
from typing import Tuple, Sequence
import jax.experimental.host_callback as hcb


def multinomial_resampling(key, lws: Sequence, trs: Sequence):
    length = len(lws)
    log_total_weight = jax.scipy.special.logsumexp(lws)
    log_normalized_weights = lws - log_total_weight
    inc_log_ml_est = jax.scipy.special.logsumexp(lws) - jnp.log(length)
    key, sub_key = jax.random.split(key)
    parents = jax.random.categorical(
        sub_key, log_normalized_weights, shape=(length,)
    )
    trs = jax.tree_util.tree_map(lambda v: v[parents], trs)
    return key, (inc_log_ml_est, trs)


def bootstrap_filter(
    model: GenerativeFunction,
    n_particles: int,
):
    def _inner(
        key,
        observation_sequence: ChoiceMap,
        model_args: Tuple,
    ):
        initial_m_args = model_args[0]
        initial_obs = observation_sequence[0]
        key, *sub_keys = jax.random.split(key, n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, (lws, trs) = jax.vmap(model.importance, in_axes=(0, None, None))(
            sub_keys,
            initial_obs,
            initial_m_args,
        )
        key, (log_ml_est, trs) = multinomial_resampling(key, lws, trs)

        # Implementing the main loop of SMC using a `jax.lax.scan`
        # significantly improves compile times, at the cost of a
        # potential loop unrolling optimization.
        #
        # In general, the generated `Jaxpr` is quite large -- so
        # the compile time speed up is significant.

        def _body_fun(carry, slice):
            key, log_ml_est, trs = carry
            obs, m_args = slice
            key, *sub_keys = jax.random.split(key, n_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, (lws, trs, d) = jax.vmap(
                model.update, in_axes=(0, 0, None, None)
            )(sub_keys, trs, obs, m_args)
            key, (inc_log_ml_est, trs) = multinomial_resampling(key, lws, trs)
            log_ml_est += inc_log_ml_est
            return (key, log_ml_est, trs), ()

        (key, log_ml_est, trs), () = jax.lax.scan(
            _body_fun,
            (key, log_ml_est, trs),
            (observation_sequence[1:], tree_stack(model_args[1:])),
        )

        return key, (trs, log_ml_est)

    return _inner


def proposal_sequential_monte_carlo(
    model: GenerativeFunction,
    initial_proposal: GenerativeFunction,
    transition_proposal: GenerativeFunction,
    n_particles: int,
):
    def _inner(
        key,
        observation_sequence: ChoiceMap,
        model_args: Sequence,
        proposal_args: Sequence,
    ):
        initial_m_args = model_args[0]
        initial_p_args = proposal_args[0]
        none_ = (None for _ in initial_p_args)
        initial_obs = observation_sequence.slice([0])
        key, *sub_keys = jax.random.split(key, n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, p_trs = jax.vmap(
            initial_proposal.simulate, in_axes=[0, (None, *none_)]
        )(
            sub_keys,
            (initial_obs, *initial_p_args),
        )

        key, *sub_keys = jax.random.split(key, n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        p_chm = p_trs.get_choices()
        chm = p_chm.merge(initial_obs)

        # This is to prevent us from having to duplicate observations
        # over all the `vmap` broadcasts.
        none_tree_obs = jtu.tree_map(lambda v: None, observation_sequence)
        none_tree_chm = jtu.tree_map(lambda v: 0, p_chm)
        none_tree = none_tree_chm.merge(none_tree_obs)

        # Here, we provide the `none_tree` from above, which should tell
        # `vmap` to not broadcast over that part of our choice tree.
        _, (lws, m_trs) = jax.vmap(
            model.importance, in_axes=(0, none_tree, None)
        )(
            sub_keys,
            chm,
            initial_m_args,
        )

        key, (log_ml_est, m_trs) = multinomial_resampling(key, lws, m_trs)

        def _body_fun(carry, slice):
            count, key, log_ml_est, m_trs = carry
            obs, m_args, p_args = slice
            key, *sub_keys = jax.random.split(key, n_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, p_trs = jax.vmap(
                transition_proposal.simulate, in_axes=(0, (0, None, *none_))
            )(sub_keys, (m_trs, obs, *p_args))
            p_chm = p_trs.get_choices()
            none_tree_chm = jtu.tree_map(lambda v: 0, p_chm)
            none_tree = IndexMask(none_tree_chm.merge(none_tree_obs), None)
            chm = IndexMask(p_chm.merge(obs), jnp.array(count))
            key, *sub_keys = jax.random.split(key, n_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, (lws, m_trs, d) = jax.vmap(
                model.update, in_axes=(0, 0, none_tree, None)
            )(sub_keys, m_trs, chm, m_args)
            lws -= p_trs.get_score()
            key, (inc_log_ml_est, trs) = multinomial_resampling(key, lws, m_trs)
            log_ml_est += inc_log_ml_est
            return (count + 1, key, log_ml_est, trs), ()

        (_, key, log_ml_est, trs), () = jax.lax.scan(
            _body_fun,
            (1, key, log_ml_est, m_trs),
            (
                observation_sequence[1:],
                tree_stack(model_args[1:]),
                tree_stack(proposal_args[1:]),
            ),
        )

        return key, (trs, log_ml_est)

    return _inner
