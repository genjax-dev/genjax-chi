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
import genjax as gex

key = jax.random.PRNGKey(314159)


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


class TestChoiceGradient:
    def test_simple_normal_gradient(self, benchmark):
        v1 = 0.5
        v2 = -0.5
        chm = gex.ChoiceMap({("y1",): v1, ("y2",): v2})
        tr = jax.jit(gex.simulate(simple_normal))(key)
        jitted = jax.jit(gex.choice_grad(simple_normal))
        choice_grads = benchmark(jitted, tr, chm, key)
        test_grad_y1 = jax.grad(lambda v1: gex.Normal().score(v1))(v1)
        test_grad_y2 = jax.grad(lambda v2: gex.Normal().score(v2))(v2)
        assert choice_grads[("y1",)] == test_grad_y1
        assert choice_grads[("y2",)] == test_grad_y2
