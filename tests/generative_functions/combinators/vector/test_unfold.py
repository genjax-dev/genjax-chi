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
import jax.tree_util as jtu
import pytest

import genjax
from genjax import diff, NoChange, UnknownChange


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
            jnp.sum(tr.project(genjax.vector_select(genjax.select("z"))))
            == unfold_score
        )

    def test_unfold_index_importance(self):
        @genjax.gen
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        model = genjax.Unfold(kernel, max_length=10)
        key = jax.random.PRNGKey(314159)
        chm = genjax.index_choice_map([3], genjax.choice_map({"z": jnp.array([0.5])}))
        key, (w, tr) = model.importance(key, chm, (6, 0.1))
        sel = genjax.index_select([3], genjax.select("z"))
        assert True

    def test_unfold_index_update(self):
        @genjax.gen
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        model = genjax.Unfold(kernel, max_length=10)
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(model))(key, (5, 0.1))
        chm = genjax.index_choice_map([3], genjax.choice_map({"z": jnp.array([0.5])}))
        key, (_, w, new_tr, _) = model.update(
            key,
            tr,
            chm,
            (genjax.diff(6, genjax.UnknownChange), genjax.diff(0.1, genjax.NoChange)),
        )
        assert new_tr.get_score() == pytest.approx(w + tr.get_score(), 0.001)

    def test_off_by_one_issue_415(self):
        key = jax.random.PRNGKey(17)

        @genjax.gen
        def one_step(_dummy_state):
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        model = genjax.Unfold(one_step, max_length=5)
        _, true_tr = model.simulate(key, (4, (0.0)))
        true_x = true_tr["x"]
        chm = genjax.vector_choice_map(
            genjax.choice_map({("x"): true_x}),
        )
        key, importance_key = jax.random.split(key)
        _, (_, importance_tr) = model.importance(importance_key, chm, (4, (0.0)))
        assert importance_tr["x"][0] == true_x[0]
        assert importance_tr["x"][1] == true_x[1]
        assert importance_tr["x"][2] == true_x[2]
        assert importance_tr["x"][3] == true_x[3]
        assert importance_tr["x"][4] == true_x[4]

    def test_update_pytree_state(self):
        @genjax.gen
        def next_step(state):
            (x_prev, z_prev) = state
            x = genjax.normal(_phi * x_prev, _q) @ "x"
            z = _beta * z_prev + x
            y = genjax.normal(z, _r) @ "y"

            return (x, z)

        key = jax.random.PRNGKey(314159)
        max_T = 20
        model = genjax.Unfold(next_step, max_length=max_T)
        model_args = (0.0, 0.0)

        def obs_chm(y, t):
            return genjax.index_choice_map(
                [t], genjax.choice_map({"y": jnp.expand_dims(y[t], 0)})
            )

        _phi, _q, _beta, _r = (0.9, 1, 0.5, 1)
        _y = jnp.array(
            [
                -0.2902139981058265,
                1.3737349808892283,
                0.18008812825243414,
                -4.916120119596394,
                -4.4309304370236084,
                -8.079005724974689,
                -6.690313781416586,
                -3.2570512312033895,
                -2.518358148235886,
                -1.6024395810401404,
                -2.9326810854675287,
                -2.1934524121301915,
                -3.1139481129728765,
                -4.297384806279307,
                -4.838146951278021,
                -4.3374767962853396,
                -4.922057827696929,
                -2.174534838472549,
                -0.9382616295106063,
                1.056841769960211,
            ]
        )

        key, init_key, update_key = jax.random.split(key, 3)
        _, (init_weight, init_tr) = model.importance(
            init_key, obs_chm(_y, 0), (0, model_args)
        )

        diffs = (
            diff(1, UnknownChange),
            jtu.tree_map(lambda v: diff(v, NoChange), model_args),
        )

        _ = model.update(update_key, init_tr, obs_chm(_y, 1), diffs)
        assert True
