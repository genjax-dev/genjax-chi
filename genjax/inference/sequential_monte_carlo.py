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
from genjax.core.datatypes import ChoiceMap, GenerativeFunction
from typing import Tuple, Sequence


def bootstrap_filter(
    model: GenerativeFunction,
    n_particles: int,
):
    pass


def proposal_sequential_monte_carlo(
    model: GenerativeFunction,
    initial_proposal: GenerativeFunction,
    transition_proposal: GenerativeFunction,
    n_particles: int,
):
    def _inner(
        key,
        model_args: Sequence[Tuple],
        proposal_args: Sequence[Tuple],
        observation_sequence: Sequence[ChoiceMap],
    ):

        # These assertions are all checked at JAX tracer time.
        assert len(model_args) > 0
        assert len(model_args) == len(proposal_args)
        assert len(proposal_args) == len(observation_sequence)
        none_ = (None for _ in proposal_args)
        iteration_length = len(model_args)
        initial_m_args = model_args[0]
        initial_p_args = proposal_args[0]
        initial_obs = observation_sequence[0]
        key, *sub_keys = jax.random.split(key, n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        _, p_trs = jax.vmap(initial_proposal, in_axes=(0, None))(
            sub_keys,
            (initial_obs, *initial_p_args),
        )
        key, *sub_keys = jax.random.split(key, n_particles + 1)
        sub_keys = jnp.array(sub_keys)
        initial_obs = jax.tree_util.map(
            lambda v: jnp.repeats(v, n_particles), initial_obs
        )
        p_chm = p_trs.get_choices()
        chm = p_chm.merge(initial_obs)
        _, (w, m_trs) = jax.vmap(model.importance, in_axes=(0, 0, None))(
            sub_keys,
            chm,
            initial_m_args,
        )
        w = w - p_trs.get_score()
        for ind in range(1, iteration_length):
            m_args = model_args[ind]
            p_args = proposal_args[ind]
            obs = observation_sequence[ind]
            obs = jax.tree_util.map(lambda v: jnp.repeats(v, n_particles), obs)
            key, *sub_keys = jax.random.split(key, n_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, sub_keys = jax.vmap(
                transition_proposal.simulate, in_axes=(0, (0, *none_))
            )(sub_keys, (p_trs, *p_args))
            p_chm = p_trs.get_choices()
            chm = p_chm.merge(obs)
            key, *sub_keys = jax.random.split(key, n_particles + 1)
            sub_keys = jnp.array(sub_keys)
            _, (new_w, m_trs, d) = jax.vmap(
                model.update, in_axes=(0, 0, 0, None)
            )(sub_keys, m_trs, chm, m_args)
            w += new_w - p_trs.get_score()
        return w, jnp.sum(w) - jnp.log(n_particles), m_trs

    return _inner
