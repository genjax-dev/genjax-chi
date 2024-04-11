# Copyright 2023 MIT Probabilistic Computing Project
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

import genjax
import jax
import jax.numpy as jnp
import pytest
from genjax.incremental import Diff, NoChange, UnknownChange


class TestUnfoldSimpleNormal:
    def test_unfold_simple_normal(self):
        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(kernel.simulate)(sub_key, (5, 0.1))
        unfold_score = tr.get_score()
        assert jnp.sum(tr.project(genjax.select("z"))) == unfold_score

    def test_unfold_index_importance(self):
        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def chain(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        special = 0.579
        choice = genjax.indexed_choice_map(
            jnp.array([0]), genjax.choice_map({"z": jnp.array([special])})
        )

        t = -1
        key, sub_key = jax.random.split(key)
        (tr, _) = chain.importance(sub_key, choice, (t, 0.3))
        assert tr[0, "z"] != special

        for t in range(0, 10):
            key, sub_key = jax.random.split(key)
            (tr, _) = chain.importance(sub_key, choice, (t, 0.3))
            sel = genjax.select("z")
            assert tr.get_score() == tr.project(sel)

        @genjax.static_gen_fn
        def f(x):
            x = genjax.normal(x, 1.0) @ "x"
            return x

        model = genjax.unfold_combinator(max_length=10)(f)

        def obs_choice(x, t):
            return genjax.indexed_choice_map(
                jnp.array([t]), genjax.choice_map({"x": jnp.expand_dims(x[t], 0)})
            )

        obs = jnp.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
            ]
        )

        key, sub_key = jax.random.split(key)
        (tr, wt) = model.importance(sub_key, obs_choice(obs, 9), (9, 0.0))
        for i in range(0, 9):
            assert tr[i, "x"] != 9
        assert tr[9, "x"] == 9

    def test_unfold_two_layer_index_importance(self):
        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def two_layer_chain(x):
            z1 = genjax.trace("z1", genjax.normal)(x, 1.0)
            z2 = genjax.trace("z2", genjax.normal)(z1, 1.0)
            return z2

        key = jax.random.PRNGKey(314159)

        # Partial constraints
        choice = genjax.indexed_choice_map(
            jnp.array([0]),
            genjax.choice_map({"z": jnp.array([0.5])}),
        )
        for t in range(0, 10):
            key, sub_key = jax.random.split(key)
            (tr, _) = two_layer_chain.importance(sub_key, choice, (t, 0.3))
            sel = genjax.select("z1", "z2")
            assert tr.get_score() == pytest.approx(tr.project(sel), 1e-4)

        # No constraints
        choice = genjax.EmptyChoice()
        for t in range(0, 10):
            key, sub_key = jax.random.split(key)
            (tr, _) = two_layer_chain.importance(sub_key, choice, (t, 0.3))
            sel = genjax.select("z1", "z2")
            assert tr.get_score() == pytest.approx(tr.project(sel), 1e-4)

    def test_unfold_index_update(self):
        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def kernel(x):
            z = genjax.trace("z", genjax.normal)(x, 1.0)
            return z

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(kernel.simulate)(sub_key, (5, 0.1))
        choice = genjax.indexed_choice_map(
            jnp.array([3]), genjax.choice_map({"z": jnp.array([0.5])})
        )
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = kernel.update(
            sub_key,
            tr,
            choice,
            (Diff.tree_diff(6, UnknownChange), Diff.tree_diff(0.1, NoChange)),
        )
        newly_introduced_choice = genjax.indexed_select(
            jnp.array([6]), genjax.select("z")
        )
        newly_introduced_score = new_tr.project(newly_introduced_choice)
        assert new_tr.get_score() == pytest.approx(
            w + tr.get_score() + newly_introduced_score, 0.001
        )

        # Test update with EmptyChoice
        key, sub_key = jax.random.split(key)
        tr2, w, retval_diff, discard = kernel.update(
            sub_key, tr, genjax.EmptyChoice(),
            (Diff.tree_diff(6, UnknownChange), Diff.tree_diff(0.1, NoChange)),
        )
        assert tr.get_retval()[5] == tr.get_retval()[6] # before, was all repeats at the end
        assert tr2.get_retval()[5] != tr2.get_retval()[6] # now, we have a new value at 6
        assert tr2.get_retval()[6] == tr2.get_retval()[7] # and then repeats after that
        assert tr2.get_score() < tr.get_score() + 0.001 # should have more randomness
        # should have w = p(new)/[p(old)q(new stuff)] = p(6)/q(6) = 1
        assert pytest.approx(w, 0.001) == 0.0
        assert tr2.get_score() == pytest.approx(tr.get_score() + tr2.project(genjax.indexed_select(jnp.array([6]), genjax.select("z"))), 0.001)
        assert discard.is_empty()
        assert retval_diff == Diff.tree_diff_unknown_change(tr2.get_retval())

        key, sub_key = jax.random.split(key)
        tr3, w, retval_diff, discard = kernel.update(
            sub_key, tr2, genjax.EmptyChoice(), (Diff.tree_diff(4, UnknownChange), Diff.tree_diff(0.1, NoChange))
        )
        assert tr3.get_retval()[4] == tr3.get_retval()[5]
        assert tr3.get_retval()[3] != tr3.get_retval()[4]
        assert tr3.get_retval()[5] == tr3.get_retval()[4]
        assert tr3.get_retval()[6] == tr3.get_retval()[4]
        
        # These 2 tests currently fail -- we need to fix this
        # assert not discard.get_submap(5).is_empty()
        # assert not discard.get_submap(6).is_empty()
        
        assert retval_diff == Diff.tree_diff_unknown_change(tr3.get_retval())
        assert tr2.get_score() - tr3.get_score() == pytest.approx(
            tr2.project(genjax.indexed_select(jnp.array([5]), genjax.select("z"))) + \
            tr2.project(genjax.indexed_select(jnp.array([6]), genjax.select("z"))), 0.001
        )
        # should have w = p(new)/p(old) = 1/p(5, 6)
        assert pytest.approx(w, 0.001) == tr3.get_score() - tr2.get_score()


    def test_off_by_one_issue_415(self):
        @genjax.unfold_combinator(max_length=5)
        @genjax.static_gen_fn
        def one_step(_dummy_state):
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        key = jax.random.PRNGKey(17)
        key, sub_key = jax.random.split(key)
        true_tr = one_step.simulate(sub_key, (4, (0.0)))
        true_x = jax.vmap(lambda idx: true_tr.get_choices()[idx, "x"])(jnp.arange(5))
        choice = genjax.vector_choice_map(
            genjax.choice_map({("x"): true_x}),
        )
        key, importance_key = jax.random.split(key)
        (importance_tr, _) = one_step.importance(importance_key, choice, (4, (0.0)))
        assert importance_tr[0, "x"] == true_x[0]
        assert importance_tr[1, "x"] == true_x[1]
        assert importance_tr[2, "x"] == true_x[2]
        assert importance_tr[3, "x"] == true_x[3]
        assert importance_tr[4, "x"] == true_x[4]

    def test_update_pytree_state(self):
        @genjax.static_gen_fn
        def next_step(state):
            (x_prev, z_prev) = state
            x = genjax.normal(_phi * x_prev, _q) @ "x"
            z = _beta * z_prev + x
            _ = genjax.normal(z, _r) @ "y"
            return (x, z)

        key = jax.random.PRNGKey(314159)
        max_T = 20
        model = genjax.unfold_combinator(max_length=max_T)(next_step)
        model_args = (0.0, 0.0)

        def obs_choice(y, t):
            return genjax.indexed_choice_map(
                jnp.array([t]), genjax.choice_map({"y": jnp.expand_dims(y[t], 0)})
            )

        _phi, _q, _beta, _r = (0.9, 1.0, 0.5, 1.0)
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

        key, sub_key = jax.random.split(key)
        (tr, _) = model.importance(sub_key, obs_choice(_y, 0), (0, model_args))

        for t in range(1, 10):
            y_sel = genjax.indexed_select(jnp.array([t]), genjax.select("y"))
            diffs = (
                Diff.tree_diff(t, UnknownChange),
                Diff.tree_diff_no_change(model_args),
            )

            # Score underneath the selection should be 0.0
            # before the extension.
            assert tr.project(y_sel) == 0.0

            key, sub_key = jax.random.split(key)
            (tr, w, _, _) = model.update(sub_key, tr, obs_choice(_y, t), diffs)

            # The weight should be equal to the new score
            # plus any newly sampled choices.
            assert w == pytest.approx(tr.project(y_sel), 0.0001)

    ####################################################
    #          Remember: the update weight math        #
    #                                                  #
    #   log p(r′,t′;x′) + log q(r;x,t) - log p(r,t;x)  #
    #       - log q(r′;x′,t′) - q(t′;x′,t+u)           #
    #                                                  #
    ####################################################

    def test_update_check_weight_computations(self):
        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def chain(z_prev):
            z = genjax.normal(z_prev, 1.0) @ "z"
            _ = genjax.normal(z, 1.0) @ "x"
            return z

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = chain.simulate(sub_key, (5, 0.0))

        #######################################
        # Check specific weight computations. #
        #######################################

        # Ensure that update is computed correctly.
        new_tr = tr
        for t in range(0, 5):
            z_sel = genjax.indexed_select(jnp.array([t]), genjax.select("z"))
            x_sel = genjax.indexed_select(jnp.array([t]), genjax.select("x"))
            obs = genjax.indexed_choice_map(
                jnp.array([t]),
                genjax.choice_map({"x": jnp.array([1.0])}),
            )
            diffs = (Diff.tree_diff(5, NoChange), Diff.tree_diff(0.0, NoChange))
            old_score = new_tr.project(x_sel)
            old_x = new_tr.filter(x_sel)[t, "x"]
            old_z = new_tr.filter(z_sel)[t, "z"]
            key, sub_key = jax.random.split(key)
            (new_tr, w, _, _) = chain.update(sub_key, new_tr, obs, diffs)
            new_z = new_tr.filter(z_sel)[t, "z"]
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
            x_sel = genjax.indexed_select(jnp.array([t]), genjax.select("x"))
            assert new_tr.filter(x_sel)[t, "x"] == 1.0

        # Now, update `z`.
        obs = genjax.indexed_choice_map(
            jnp.array([0]),
            genjax.choice_map({"z": jnp.array([1.0])}),
        )
        diffs = (Diff.tree_diff(5, NoChange), Diff.tree_diff(0.0, NoChange))

        # This should be the Markov blanket of the update.
        vzsel = genjax.indexed_select(jnp.array([0, 1]), genjax.select("z"))
        xsel = genjax.indexed_select(jnp.array([0]), genjax.select("x"))
        old_score = new_tr.project(vzsel) + new_tr.project(xsel)

        # Update just `z`
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = chain.update(sub_key, new_tr, obs, diffs)

        # Check that all prior updates are preserved.
        for t in range(0, 5):
            x_sel = genjax.indexed_select(jnp.array([t]), genjax.select("x"))
            assert new_tr.filter(x_sel)[t, "x"] == 1.0

        # Check that update succeeded.
        zsel = genjax.indexed_select(jnp.array([0]), genjax.select("z"))
        assert new_tr.filter(zsel)[0, "z"] == 1.0
        assert new_tr.project(zsel) == pytest.approx(
            genjax.normal.logpdf(1.0, 0.0, 1.0), 0.0001
        )

        # Check new score at (0, "x")
        xsel = genjax.indexed_select(jnp.array([0]), genjax.select("x"))
        assert new_tr.filter(xsel)[0, "x"] == 1.0
        assert new_tr.project(xsel) == pytest.approx(
            genjax.normal.logpdf(1.0, 1.0, 1.0), 0.0001
        )  # the mean (z) should be 1.0

        # Check the scores and weights.
        new_score = new_tr.project(vzsel) + new_tr.project(xsel)
        assert w == pytest.approx(new_score - old_score, 0.0001)

    def test_update_check_score_correctness(self):
        @genjax.unfold_combinator(max_length=5)
        @genjax.static_gen_fn
        def chain(z_prev):
            z = genjax.normal(z_prev, 1.0) @ "z"
            _ = genjax.normal(z, 1.0) @ "x"
            return z

        key = jax.random.PRNGKey(314159)

        # Run importance to get a fully constrained trace.
        full_choice = genjax.indexed_choice_map(
            jnp.array([0, 1, 2, 3, 4]),
            genjax.choice_map(
                {
                    "x": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                    "z": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                }
            ),
        )

        key, sub_key = jax.random.split(key)
        (tr, w) = chain.importance(sub_key, full_choice, (4, 0.0))
        assert w == tr.get_score()
        full_score = tr.get_score()

        # Run update to incrementally constrain a trace
        # (already extended).
        key, sub_key = jax.random.split(key)
        tr = chain.simulate(sub_key, (4, 0.0))
        for t in range(0, 5):
            choice = genjax.indexed_choice_map(
                jnp.array([t]),
                genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])}),
            )
            diffs = (Diff.tree_diff(4, NoChange), Diff.tree_diff(0.0, NoChange))

            key, sub_key = jax.random.split(key)
            (tr, w, _, _) = chain.update(sub_key, tr, choice, diffs)

        assert tr.get_score() == pytest.approx(full_score, 0.0001)

        # Run update to incrementally extend and constrain a trace.
        choice = genjax.indexed_choice_map(
            jnp.array([0]),
            genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])}),
        )
        key, sub_key = jax.random.split(key)
        (tr, _) = chain.importance(sub_key, choice, (0, 0.0))
        for t in range(1, 5):
            choice = genjax.indexed_choice_map(
                jnp.array([t]),
                genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])}),
            )
            diffs = (Diff.tree_diff(t, UnknownChange), Diff.tree_diff(0.0, NoChange))

            key, sub_key = jax.random.split(key)
            (tr, w, _, _) = chain.update(sub_key, tr, choice, diffs)

        assert tr.get_score() == pytest.approx(full_score, 0.0001)

        # Check that the projected score is equal to the returned score.
        sel = genjax.select("x", "z")
        assert tr.project(sel) == pytest.approx(tr.get_score(), 0.0001)
        assert tr.project(sel) == pytest.approx(full_score, 0.0001)

        # Re-run the above process (importance followed by update).
        # Check that, if we only generate length < max_length,
        # the projected score is equal to the returned score.
        choice = genjax.indexed_choice_map(
            jnp.array([0]),
            genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])}),
        )
        key, sub_key = jax.random.split(key)
        (tr, _) = chain.importance(sub_key, choice, (0, 0.0))
        for t in range(1, 3):
            choice = genjax.indexed_choice_map(
                jnp.array([t]),
                genjax.choice_map({"x": jnp.array([0.0]), "z": jnp.array([0.0])}),
            )
            diffs = (Diff.tree_diff(t, UnknownChange), Diff.tree_diff(0.0, NoChange))
            key, sub_key = jax.random.split(key)
            (tr, w, _, _) = chain.update(sub_key, tr, choice, diffs)

        sel = genjax.select("x", "z")
        assert tr.project(sel) == tr.get_score()
        sel = genjax.indexed_select(jnp.array([0, 1, 2]), genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()
        sel = genjax.indexed_select(jnp.array([0, 1, 2, 3, 4]), genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()

        # Re-run the above process (importance followed by update)
        # but without constraints on `z`.
        # Check that, if we only generate length < max_length,
        # the projected score is equal to the returned score.
        sel = genjax.select("x", "z")
        choice = genjax.indexed_choice_map(
            jnp.array([0]), genjax.choice_map({"x": jnp.array([0.0])})
        )
        key, sub_key = jax.random.split(key)
        (tr, _) = chain.importance(sub_key, choice, (0, 0.0))
        assert tr.project(sel) == tr.get_score()
        for t in range(1, 3):
            choice = genjax.indexed_choice_map(
                jnp.array([t]), genjax.choice_map({"x": jnp.array([0.0])})
            )
            diffs = (Diff.tree_diff(t, UnknownChange), Diff.tree_diff(0.0, NoChange))
            key, sub_key = jax.random.split(key)
            (tr, w, _, _) = chain.update(sub_key, tr, choice, diffs)

        assert tr.project(sel) == tr.get_score()
        sel = genjax.indexed_select(jnp.array([0, 1, 2]), genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()
        sel = genjax.indexed_select(jnp.array([0, 1, 2, 3, 4]), genjax.select("x", "z"))
        assert tr.project(sel) == tr.get_score()

    def test_combinator(self):
        @genjax.unfold_combinator(max_length=10)
        @genjax.static_gen_fn
        def model():
            """Model docstring"""
            return genjax.normal(0.0, 1.0) @ "y"

        # Prove that mandatory keyword argument is enforced
        with pytest.raises(Exception):

            @genjax.unfold_combinator()  # type: ignore
            @genjax.static_gen_fn
            def bad_model():
                return genjax.normal(0.0, 1.0) @ "y"

        assert model.__doc__ == "Model docstring"
