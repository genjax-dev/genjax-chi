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

"""This module supports a set of (WIP) integration interfaces with variants of
Hamiltonian Monte Carlo exported by the :code:`blackjax` sampling library.

[Blackjax]_.

.. note::

    .. [Blackjax] BlackJAX is a sampling library designed for ease of use, speed and modularity. (https://github.com/blackjax-devs/blackjax)
"""

from typing import Union

import blackjax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax.core import ChoiceMap
from genjax.core import Selection
from genjax.core import Trace


Int = Union[jnp.ndarray, np.ndarray]


def _estimate(trace: Trace, key, chm: ChoiceMap):
    gen_fn = trace.get_gen_fn()
    args = trace.get_args()
    key, (w, _, _) = gen_fn.update(key, trace, chm, args)
    return w


def hamiltonian_monte_carlo(
    trace: Trace, selection: Selection, num_steps: Int, *args
):
    def _one_step(kernel, state, key):
        state, _ = kernel(key, state)
        return state, state

    def _inner(key, **kwargs):
        def logprob(chm):
            return _estimate(trace, key, chm)

        hmc = blackjax.hmc(logprob, *args, **kwargs)
        initial_position, _ = selection.filter(trace)
        initial_position = initial_position.strip()
        stripped = jtu.tree_map(
            lambda v: v if v.dtype == jnp.float32 else None,
            initial_position,
        )
        initial_state = hmc.init(stripped)

        def step(state, key):
            return _one_step(hmc.step, state, key)

        key, *sub_keys = jax.random.split(key, num_steps + 1)
        sub_keys = jnp.array(sub_keys)
        _, states = jax.lax.scan(step, initial_state, sub_keys)
        final_positions = jtu.tree_map(
            lambda a, b: a if b is None else b,
            initial_position,
            states.position,
        )
        return key, final_positions

    return _inner


def no_u_turn_sampler(
    trace: Trace, selection: Selection, num_steps: Int, *args
):
    def _one_step(kernel, state, key):
        state, _ = kernel(key, state)
        return state, state

    def _inner(key, **kwargs):
        def logprob(chm):
            return _estimate(trace, key, chm)

        hmc = blackjax.nuts(logprob, *args, **kwargs)
        initial_position, _ = selection.filter(trace)
        initial_position = initial_position.strip()
        stripped = jtu.tree_map(
            lambda v: v if v.dtype == jnp.float32 else None,
            initial_position,
        )
        initial_state = hmc.init(stripped)

        def step(state, key):
            return _one_step(hmc.step, state, key)

        key, *sub_keys = jax.random.split(key, num_steps + 1)
        sub_keys = jnp.array(sub_keys)
        _, states = jax.lax.scan(step, initial_state, sub_keys)
        final_positions = jtu.tree_map(
            lambda a, b: a if b is None else b,
            initial_position,
            states.position,
        )
        return key, final_positions

    return _inner


hmc = hamiltonian_monte_carlo
nuts = no_u_turn_sampler
