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

import dataclasses

import blackjax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax.core import ChoiceMap
from genjax.core import Selection
from genjax.core import Trace
from genjax.core.diff_rules import Diff
from genjax.core.typing import Int
from genjax.core.typing import PRNGKey
from genjax.inference.mcmc.kernel import MCMCKernel


# Uses incremental computing to prune non-contributing sites in update.
def _estimate(key: PRNGKey, trace: Trace, chm: ChoiceMap):
    args = trace.get_args()

    # Here, the arguments (and hence, the target) don't change
    # if they do change, you shouldn't be using MCMC...because
    # then the target is changing.
    argdiffs = tuple(map(Diff.no_change, args))

    key, (_, _, updated_trace, _) = trace.update(
        key,
        trace,
        chm,
        argdiffs,
    )
    return updated_trace.get_score()


@dataclasses.dataclass
class HamiltonianMonteCarlo(MCMCKernel):
    selection: Selection
    num_steps: Int

    def flatten(self):
        return (), (self.selection, self.num_steps)

    def apply(self, key: PRNGKey, trace: Trace):
        def _one_step(kernel, state, key):
            state, _ = kernel(key, state)
            return state, state

        # TODO: assess the validity of using `update` to compute
        # new logpdf values.
        key, incremental_logpdf = trace.get_incremental_logpdf(key)

        hmc = blackjax.hmc(incremental_logpdf)
        initial_position, _ = self.selection.filter(trace.strip())
        stripped = jtu.tree_map(
            lambda v: v if v.dtype == jnp.float32 else None,
            initial_position,
        )
        initial_state = hmc.init(stripped)

        def step(state, key):
            return _one_step(hmc.step, state, key)

        key, *sub_keys = jax.random.split(key, self.num_steps + 1)
        sub_keys = jnp.array(sub_keys)
        _, states = jax.lax.scan(step, initial_state, sub_keys)
        final_positions = jtu.tree_map(
            lambda a, b: a if b is None else b,
            initial_position,
            states.position,
        )
        return key, final_positions

    def reversal(self):
        return self


@dataclasses.dataclass
class NoUTurnSampler(MCMCKernel):
    selection: Selection
    num_steps: Int

    def flatten(self):
        return (), (self.selection, self.num_steps)

    def apply(self, key: PRNGKey, trace: Trace):
        def _one_step(kernel, state, key):
            state, _ = kernel(key, state)
            return state, state

        # TODO: assess the validity of using `update` to compute
        # new logpdf values.
        key, incremental_logpdf = trace.get_incremental_logpdf(key)

        hmc = blackjax.nuts(incremental_logpdf)
        initial_position, _ = self.selection.filter(trace.strip())
        stripped = jtu.tree_map(
            lambda v: v if v.dtype == jnp.float32 else None,
            initial_position,
        )
        initial_state = hmc.init(stripped)

        def step(state, key):
            return _one_step(hmc.step, state, key)

        key, *sub_keys = jax.random.split(key, self.num_steps + 1)
        sub_keys = jnp.array(sub_keys)
        _, states = jax.lax.scan(step, initial_state, sub_keys)
        final_positions = jtu.tree_map(
            lambda a, b: a if b is None else b,
            initial_position,
            states.position,
        )
        return key, final_positions

    def reversal(self):
        return self


##############
# Shorthands #
##############

hmc = HamiltonianMonteCarlo
nuts = NoUTurnSampler
