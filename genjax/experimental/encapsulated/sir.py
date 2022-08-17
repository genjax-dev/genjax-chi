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
Sampling Importance Resampling (SIR) as an `EncapsulatedGenerativeFunction`.
"""

from dataclasses import dataclass
from genjax.encapsulated import EncapsulatedGenerativeFunction
from genjax.generative_function import simulate, importance
import jax
import jax.numpy as jnp
from jax._src import abstract_arrays
from typing import Callable


@dataclass
class EncapsulatedSIR(EncapsulatedGenerativeFunction):
    particles: int
    model: Callable
    proposal: Callable

    # Define `Pytree` methods.
    def flatten(self):
        return self.particles, (self.initial_proposal, self.transition_proposal)

    def unflatten(self, data, xs):
        return EncapsulatedSIR(*xs, *data)

    # Define abstract evaluation.
    def abstract_eval(self, key, *args, **kwargs):
        return key, abstract_arrays.ShapedArray(shape=(), dtype=bool)

    # Define GFI methods.
    def simulate(self, key, args):
        observations = args[0]
        model_args = args[1]
        proposal_args = args[2]
        key, subkeys = jax.random.split(key, self.particles + 1)
        subkeys = jnp.array(subkeys)
        key, trs = jax.vmap(simulate(self.proposal), in_axes=(0, None))(
            subkeys, proposal_args
        )
        chms = jax.vmap(lambda tr: observations.merge(tr.get_choices()))(trs)
        key, subkeys = jax.random.split(key, self.particles + 1)
        subkeys = jnp.array(subkeys)
        key, (fwd_weights, _) = jax.vmap(
            importance(self.model), in_axes=(0, 0, None)
        )(subkeys, chms, model_args)
        return key, tr

    def importance(self, args):
        pass
