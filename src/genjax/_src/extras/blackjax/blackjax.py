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

from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import Trace
from genjax._src.core.pytree import tree_grad_split
from genjax._src.core.pytree import tree_zipper
from genjax._src.core.typing import Any
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.inference.mcmc.kernel import MCMCKernel


@dataclasses.dataclass
class HamiltonianMonteCarlo(MCMCKernel):
    selection: Selection
    step_size: Any
    inverse_mass_matrix: Any
    num_integration_steps: Int
    num_steps: Int

    def flatten(self):
        return (), (
            self.selection,
            self.step_size,
            self.inverse_mass_matrix,
            self.num_integration_steps,
            self.num_steps,
        )

    def apply(self, key: PRNGKey, trace: Trace):
        def _one_step(kernel, state, key):
            state, _ = kernel(key, state)
            return state, state

        # We grab the logpdf (`assess`) interface method,
        # specialize it on the arguments - because the inference target
        # is not changing. The only update which can occur is to
        # the choice map.
        gen_fn = trace.get_gen_fn()
        fixed = self.selection.complement().filter(trace.strip())
        initial_chm_position = self.selection.filter(trace.strip())
        key, scorer, _ = gen_fn.unzip(key, fixed)

        # These go into the gradient interfaces.
        grad, nograd = tree_grad_split(
            (initial_chm_position, trace.get_args()),
        )

        # The nograd component never changes.
        def _logpdf(grad):
            return scorer(grad, nograd)

        hmc = blackjax.hmc(
            _logpdf,
            self.step_size,
            self.inverse_mass_matrix,
            self.num_integration_steps,
        )

        # Pass the grad component into the HMC init.
        initial_state = hmc.init(grad)

        def step(state, key):
            return _one_step(hmc.step, state, key)

        # TODO: do we need to allocate keys for the full chain?
        # Shouldn't it just pass a single key along?
        key, *sub_keys = jax.random.split(key, self.num_steps + 1)
        sub_keys = jnp.array(sub_keys)
        _, states = jax.lax.scan(step, initial_state, sub_keys)
        final_positions, _ = tree_zipper(states.position, nograd)
        return key, final_positions

    def reversal(self):
        return self


@dataclasses.dataclass
class NoUTurnSampler(MCMCKernel):
    selection: Selection
    step_size: Any
    inverse_mass_matrix: Any
    num_steps: Int

    def flatten(self):
        return (), (
            self.selection,
            self.step_size,
            self.inverse_mass_matrix,
            self.num_steps,
        )

    def apply(self, key: PRNGKey, trace: Trace):
        def _one_step(kernel, state, key):
            state, _ = kernel(key, state)
            return state, state

        # We grab the logpdf (`assess`) interface method,
        # specialize it on the arguments - because the inference target
        # is not changing. The only update which can occur is to
        # the choice map.
        gen_fn = trace.get_gen_fn()
        fixed = self.selection.complement().filter(trace.strip())
        initial_chm_position = self.selection.filter(trace.strip())
        key, scorer, _ = gen_fn.unzip(key, fixed)

        # These go into the gradient interfaces.
        grad, nograd = tree_grad_split(
            (initial_chm_position, trace.get_args()),
        )

        # The nograd component never changes.
        def _logpdf(grad):
            return scorer(grad, nograd)

        hmc = blackjax.nuts(
            _logpdf,
            self.step_size,
            self.inverse_mass_matrix,
        )

        # Pass the grad component into the HMC init.
        initial_state = hmc.init(grad)

        def step(state, key):
            return _one_step(hmc.step, state, key)

        # TODO: do we need to allocate keys for the full chain?
        # Shouldn't it just pass a single key along?
        key, *sub_keys = jax.random.split(key, self.num_steps + 1)
        sub_keys = jnp.array(sub_keys)
        _, states = jax.lax.scan(step, initial_state, sub_keys)
        final_positions, _ = tree_zipper(states.position, nograd)
        return key, final_positions

    def reversal(self):
        return self


##############
# Shorthands #
##############

hmc = HamiltonianMonteCarlo.new
nuts = NoUTurnSampler.new
