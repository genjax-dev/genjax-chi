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
from model_config import hidden_markov_model

import genjax


console = genjax.go_pretty()

key = jax.random.PRNGKey(314159)
num_steps = 20
config = genjax.DiscreteHMMConfiguration.new(20, 1, 1, 0.2, 0.05)
key, tr = jax.jit(genjax.simulate(hidden_markov_model))(
    key, (num_steps, config)
)
observation_sequence = tr["z", "observation"][0:20]
key, tr = jax.jit(genjax.DiscreteHMM.simulate)(
    key, (config, observation_sequence)
)
console.print(observation_sequence)
console.print(tr.get_choices().get_leaf_value())
