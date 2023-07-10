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
import jax.numpy as jnp

import genjax


class TestUnfoldSimpleNormal:
    def test_unfold_simple_normal(self):
        @genjax.gen
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        model = genjax.Unfold(kernel, max_length=10)
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(model))(key, (5, 0.1))
        unfold_score = tr.get_score()
        assert (
            jnp.sum(
                tr.project(genjax.vector_select(jnp.arange(0, 6), genjax.select("z")))
            )
            == unfold_score
        )
    
    def test_off_by_one_issue_415(self):
        key = jax.random.PRNGKey(17)

        @genjax.gen
        def one_step(_dummy_state):
            x = genjax.normal(0., 1.) @ 'x'
            return x

        model = genjax.Unfold(one_step, max_length=5)
        _, true_tr = model.simulate(key, (4, (0.)))
        true_x = true_tr['x']
        chm = genjax.vector_choice_map(
            jnp.arange(0, len(true_x)), genjax.choice_map({('x'): true_x}))
        key, importance_key = jax.random.split(key)
        _, (_, importance_tr) = model.importance(importance_key, chm, (0, (0.)))
        assert importance_tr['x'][0] == true_x[0]
        assert importance_tr['x'][1] != true_x[1]
        assert importance_tr['x'][2] != true_x[2]
        assert importance_tr['x'][3] != true_x[3]
        assert importance_tr['x'][4] != true_x[4]
