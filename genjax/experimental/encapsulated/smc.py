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
Sequential Monte Carlo (SMC) as an `EncapsulatedGenerativeFunction`.
"""

from dataclasses import dataclass
from genjax.encapsulated import EncapsulatedGenerativeFunction
from genjax.generative_function import simulate
from jax._src import abstract_arrays
from typing import Callable


@dataclass
class EncapsulatedSMC(EncapsulatedGenerativeFunction):
    particles: int
    target_model: Callable
    initial_proposal: Callable
    transition_proposal: Callable

    # Define `Pytree` methods.
    def flatten(self):
        return self.particles, (self.initial_proposal, self.transition_proposal)

    def unflatten(self, data, xs):
        return EncapsulatedSMC(*xs, *data)

    # Define abstract evaluation.
    def abstract_eval(self, key, *args, **kwargs):
        return key, abstract_arrays.ShapedArray(shape=(), dtype=bool)

    # Define GFI methods.
    def simulate(self, key, args):
        observations = args[0]
        model_args = args[1]
        proposal_args = args[2]
        key, tr = simulate(self.initial_proposal)(key, args)
        return key, tr

    def importance(self, args):
        pass
