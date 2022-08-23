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

key = jax.random.PRNGKey(314159)


@genjax.gen
def simple_normal(key):
    key, y1 = genjax.trace("y1", genjax.Normal)(key)
    key, y2 = genjax.trace("y2", genjax.Normal)(key)
    return key, y1 + y2


def score(v1):
    _, (w, _) = genjax.Normal.importance(key, genjax.ValueChoiceMap(v1), ())
    return w


class TestChoiceGradient:
    def test_simple_normal_gradient(self, benchmark):
        v1 = 0.5
        v2 = -0.5
        chm = genjax.ChoiceMap({("y1",): v1, ("y2",): v2})
        new_key, tr = jax.jit(genjax.simulate(simple_normal))(key, ())
        jitted = jax.jit(genjax.choice_grad(simple_normal))
        new_key, choice_grads = benchmark(jitted, new_key, tr, chm, ())
        test_grad_y1 = jax.grad(score)(v1)
        test_grad_y2 = jax.grad(score)(v2)
        assert choice_grads[("y1",)].get_value() == test_grad_y1
        assert choice_grads[("y2",)].get_value() == test_grad_y2
