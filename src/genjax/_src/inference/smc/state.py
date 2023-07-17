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

from dataclasses import dataclass

from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray


@dataclass
class SMCState(Pytree):
    n_particles: IntArray
    particles: Trace
    log_weights: FloatArray
    log_ml_est: FloatArray
    valid: BoolArray

    def flatten(self):
        return (self.particles, self.log_weights, self.log_ml_est, self.valid), (
            self.n_particles,
        )

    def get_target_gen_fn(self):
        return self.particles.get_gen_fn()

    def get_particles(self):
        return self.particles

    def get_num_particles(self):
        return self.n_particles
