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
        @genjax.gen(genjax.Unfold, max_length=10)
        def chain(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        chm = genjax.index_choice_map([0], genjax.choice_map({"z": jnp.array([0.5])}))
        for t in range(0, 10):
            key, (_, tr) = chain.importance(key, chm, (t, 0.3))
            sel = genjax.vector_select("z")
            assert tr.get_score() == tr.project(sel)

    def test_unfold_two_layer_index_importance(self):
        @genjax.gen(genjax.Unfold, max_length=10)
        def two_layer_chain(x):
            z1 = genjax.trace("z1", genjax.normal)(x, 1.0)
            z2 = genjax.trace("z2", genjax.normal)(z1, 1.0)
            return z2

        key = jax.random.PRNGKey(314159)

        # Partial constraints
        chm = genjax.index_choice_map(
            [0],
            genjax.choice_map({"z": jnp.array([0.5])}),
        )
        for t in range(0, 10):
            key, (_, tr) = two_layer_chain.importance(key, chm, (t, 0.3))
            sel = genjax.vector_select("z1", "z2")
            assert tr.get_score() == tr.project(sel)

        # No constraints
        chm = genjax.EmptyChoiceMap()
        for t in range(0, 10):
            key, (_, tr) = two_layer_chain.importance(key, chm, (t, 0.3))
            sel = genjax.vector_select("z1", "z2")
            assert tr.get_score() == tr.project(sel)

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
            _ = genjax.normal(z, _r) @ "y"
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

        key, (_, tr) = model.importance(key, obs_chm(_y, 0), (0, model_args))

        for t in range(1, 10):
            y_sel = genjax.index_select([t], genjax.select("y"))
            x_sel = genjax.index_select([t], genjax.select("x"))
            diffs = (
                diff(t, UnknownChange),
                jtu.tree_map(lambda v: diff(v, NoChange), model_args),
            )

            # Score underneath the selection should be 0.0
            # before the extension.
            assert tr.project(y_sel) == 0.0

            key, (_, w, tr, _) = model.update(key, tr, obs_chm(_y, t), diffs)

            # The weight should be equal to the new score
            # plus any newly sampled choices.
            assert w == tr.project(y_sel) + tr.project(x_sel)

    def test_update_check_weight_computations(self):
        @genjax.gen(genjax.Unfold, max_length=10)
        def chain(z_prev):
            z = genjax.normal(z_prev, 1.0) @ "z"
            _ = genjax.normal(z, 1.0) @ "x"
            return z

        key = jax.random.PRNGKey(314159)
        key, tr = chain.simulate(key, (5, 0.0))

        #####
        # Check specific weight computations.
        #####

        # Ensure that update is computed correctly.
        new_tr = tr
        for t in range(0, 5):
            z_sel = genjax.index_select([t], genjax.select("z"))
            x_sel = genjax.index_select([t], genjax.select("x"))
            obs = genjax.index_choice_map(
                [t],
                genjax.choice_map({"x": jnp.array([1.0])}),
            )
            diffs = (genjax.diff(5, NoChange), genjax.diff(0.0, NoChange))
            old_score = new_tr.project(x_sel)
            old_x = x_sel.filter(new_tr.strip())["x"]
            old_z = z_sel.filter(new_tr.strip())["z"]
            key, (_, w, new_tr, _) = chain.update(key, new_tr, obs, diffs)
            new_z = z_sel.filter(new_tr.strip())["z"]
            assert old_z == new_z
            assert new_tr.project(x_sel) == pytest.approx(
                genjax.normal.logpdf(1.0, new_z, 1.0), 0.0001
            )
            assert w == pytest.approx(
                genjax.normal.logpdf(1.0, old_z, 1.0)
                - genjax.normal.logpdf(old_x, old_z, 1.0),
                0.0001,
            )

        # Check that all prior updates are preserved
        # over subsequent calls.
        for t in range(0, 5):
            x_sel = genjax.index_select([t], genjax.select("x"))
            assert x_sel.filter(new_tr.strip())["x"] == 1.0

        # Now, update `z`.
        obs = genjax.index_choice_map(
            [0],
            genjax.choice_map({"z": jnp.array([1.0])}),
        )
        diffs = (genjax.diff(5, NoChange), genjax.diff(0.0, NoChange))

        # This should be the Markov blanket of the update.
        vzsel = genjax.index_select([0, 1], genjax.select("z"))
        xsel = genjax.index_select([0], genjax.select("x"))
        old_score = new_tr.project(vzsel) + new_tr.project(xsel)

        # Update just `z`
        key, (_, w, new_tr, _) = chain.update(key, new_tr, obs, diffs)

        # Check that all prior updates are preserved.
        for t in range(0, 5):
            x_sel = genjax.index_select([t], genjax.select("x"))
            assert x_sel.filter(new_tr.strip())["x"] == 1.0

        # Check that update succeeded.
        zsel = genjax.index_select([0], genjax.select("z"))
        assert zsel.filter(new_tr.strip())["z"] == 1.0
        assert new_tr.project(zsel) == pytest.approx(
            genjax.normal.logpdf(1.0, 0.0, 1.0), 0.0001
        )

        # Check new score at (0, "x")
        xsel = genjax.index_select([0], genjax.select("x"))
        assert xsel.filter(new_tr.strip())["x"] == 1.0
        assert new_tr.project(xsel) == pytest.approx(
            genjax.normal.logpdf(1.0, 1.0, 1.0), 0.0001
        )  # the mean (z) should be 1.0

        # Check the scores and weights.
        new_score = new_tr.project(vzsel) + new_tr.project(xsel)
        assert w == pytest.approx(new_score - old_score, 0.0001)

    def test_update_check_score_correctness(self):
        @genjax.gen(genjax.Unfold, max_length=5)
        def chain(z_prev):
            z = genjax.normal(z_prev, 1.0) @ "z"
            _ = genjax.normal(z, 1.0) @ "x"
            return z

        key = jax.random.PRNGKey(314159)

        # Run importance to get a fully constrained trace.
        full_chm = genjax.index_choice_map(
            [0, 1, 2, 3, 4],
            genjax.choice_map(
                {
                    "x": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                    "z": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                }
            ),
        )

        key, (w, tr) = chain.importance(key, full_chm, (4, 0.0))
        assert w == tr.get_score()
        full_score = tr.get_score()

        # Run update to incrementally constrain a trace
        # (already extended).
        key, tr = chain.simulate(key, (4, 0.0))
        for t in range(0, 5):
            chm = genjax.index_choice_map(
                [t], genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])})
            )
            diffs = (diff(4, NoChange), diff(0.0, NoChange))

            key, (_, w, tr, _) = chain.update(key, tr, chm, diffs)

        assert tr.get_score() == pytest.approx(full_score, 0.0001)

        # Run update to incrementally extend and constrain a trace.
        chm = genjax.index_choice_map(
            [0], genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])})
        )
        key, (_, tr) = chain.importance(key, chm, (0, 0.0))
        for t in range(1, 5):
            chm = genjax.index_choice_map(
                [t], genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])})
            )
            diffs = (diff(t, UnknownChange), diff(0.0, NoChange))

            key, (_, w, tr, _) = chain.update(key, tr, chm, diffs)

        assert tr.get_score() == pytest.approx(full_score, 0.0001)

        # Check that the projected score is equal to the returned score.
        sel = genjax.vector_select("x", "z")
        assert tr.project(sel) == tr.get_score()
        assert tr.project(sel) == pytest.approx(full_score, 0.0001)

        # Re-run the above process (importance followed by update).
        # Check that, if we only generate length < max_length,
        # the projected score is equal to the returned score.
        chm = genjax.index_choice_map(
            [0], genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])})
        )
        key, (_, tr) = chain.importance(key, chm, (0, 0.0))
        for t in range(1, 3):
            chm = genjax.index_choice_map(
                [t], genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])})
            )
            diffs = (diff(t, UnknownChange), diff(0.0, NoChange))

            key, (_, w, tr, _) = chain.update(key, tr, chm, diffs)

        sel = genjax.vector_select("x", "z")
        assert tr.project(sel) == tr.get_score()
        sel = genjax.index_select([0, 1, 2], genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()
        sel = genjax.index_select([0, 1, 2, 3, 4], genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()

        # Re-run the above process (importance followed by update)
        # but without constraints on `z`.
        # Check that, if we only generate length < max_length,
        # the projected score is equal to the returned score.
        sel = genjax.vector_select("x", "z")
        chm = genjax.index_choice_map([0], genjax.choice_map({"x": jnp.array([0.0])}))
        key, (_, tr) = chain.importance(key, chm, (0, 0.0))
        assert tr.project(sel) == tr.get_score()
        for t in range(1, 3):
            chm = genjax.index_choice_map(
                [t], genjax.choice_map({"x": jnp.array([0.0])})
            )
            diffs = (diff(t, UnknownChange), diff(0.0, NoChange))

            key, (_, w, tr, _) = chain.update(key, tr, chm, diffs)

        assert tr.project(sel) == tr.get_score()
        sel = genjax.index_select([0, 1, 2], genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()
        sel = genjax.index_select([0, 1, 2, 3, 4], genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()
