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

import genjax


class TestDynamicAddresses:
    def test_dynamic_addresses_and_interfaces(self):
        @genjax.gen
        def simple_normal(index: genjax.typing.IntArray):
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace(index, genjax.normal)(0.0, 1.0)
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        tr = simple_normal.simulate(key, (3,))

    def test_basic_smc_pattern(self):
        @genjax.gen(genjax.Unfold, max_length=10)
        def chain(z):
            new_z = genjax.tfp_normal(0.0, 1.0) @ "z"
            new_x = genjax.tfp_normal(new_z, 1.0) @ "x"
            return new_z

        @genjax.gen
        def temporal_proposal(obs, t):
            obs_at_t = obs[t, "x"]
            new_z = genjax.tfp_normal(obs_at_t, 1.0) @ (t, "z")

        key = jax.random.PRNGKey(314159)
        dynamic_constraints = temporal_proposal.propose(key, )
