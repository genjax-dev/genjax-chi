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
import pytest

import genjax


class TestStaticAddressChecks:
    def test_simple_normal_addr_dup(self):
        @genjax.gen(genjax.Static)
        def simple_normal_addr_dup():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y1"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(Exception):
            _ = genjax.simulate(simple_normal_addr_dup)(key, ())

    def test_simple_normal_addr_tracer(self):
        @genjax.gen(genjax.Static)
        def simple_normal_addr_tracer():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ y1
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(Exception):
            _ = genjax.simulate(simple_normal_addr_tracer)(key, ())
