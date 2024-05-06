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

from typing import Any

import genjax
import jax
import jax.numpy as jnp
import pytest
from genjax import ChoiceMap as C
from genjax import Diff, Pytree, RemoveSelectionUpdateSpec
from genjax.generative_functions.static import AddressReuse, StaticAddressJAX
from genjax.typing import FloatArray
from jax._src.interpreters.partial_eval import DynamicJaxprTracer

#############
# Datatypes #
#############

##################################
# Generative function interfaces #
##################################


class TestStaticGenFnSimulate:
    def test_simulate_with_no_choices(self):
        @genjax.static_gen_fn
        def empty(x):
            return jnp.square(x - 3.0)

        key = jax.random.PRNGKey(314159)
        fn = jax.jit(empty.simulate)
        key, sub_key = jax.random.split(key)
        tr = fn(sub_key, (jnp.ones(4),))
        assert tr.get_score() == 0.0

    def test_simple_normal_simulate(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        fn = jax.jit(simple_normal.simulate)
        key, sub_key = jax.random.split(key)
        tr = fn(sub_key, ())
        choice = tr.get_sample()
        (_, score1, _) = genjax.normal.importance(
            key, choice.get_submap("y1"), (0.0, 1.0)
        )
        (_, score2, _) = genjax.normal.importance(
            key, choice.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_multiple_returns(self):
        @genjax.static_gen_fn
        def simple_normal_multiple_returns():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        fn = jax.jit(simple_normal_multiple_returns.simulate)
        tr = fn(sub_key, ())
        y1_ = tr.get_sample()["y1"]
        y2_ = tr.get_sample()["y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (score1, _) = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        (score2, _) = genjax.normal.assess(C.v(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_hierarchical_simple_normal_multiple_returns(self):
        @genjax.static_gen_fn
        def _submodel():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1, y2

        @genjax.static_gen_fn
        def hierarchical_simple_normal_multiple_returns():
            y1, y2 = _submodel() @ "y1"
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        fn = jax.jit(hierarchical_simple_normal_multiple_returns.simulate)
        tr = fn(sub_key, ())
        y1_ = tr.get_sample()["y1", "y1"]
        y2_ = tr.get_sample()["y1", "y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (score1, _) = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        (score2, _) = genjax.normal.assess(C.v(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)


class TestStaticGenFnAssess:
    def test_assess_with_no_choices(self):
        @genjax.static_gen_fn
        def empty(x):
            return jnp.square(x - 3.0)

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(empty.simulate)(sub_key, (jnp.ones(4),))
        jitted = jax.jit(empty.assess)
        chm = tr.get_sample()
        (score, retval) = jitted(chm, (jnp.ones(4),))
        assert score == tr.get_score()

    def test_simple_normal_assess(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_normal.assess)
        choice = tr.get_sample()
        (score, retval) = jitted(choice, ())
        assert score == tr.get_score()


@Pytree.dataclass
class CustomTree(genjax.Pytree):
    x: Any
    y: Any


@genjax.static_gen_fn
def simple_normal(custom_tree):
    y1 = genjax.normal(custom_tree.x, 1.0) @ "y1"
    y2 = genjax.normal(custom_tree.y, 1.0) @ "y2"
    return CustomTree(y1, y2)


@Pytree.dataclass
class _CustomNormal(genjax.Distribution):
    def estimate_logpdf(self, key, v, custom_tree):
        w, _ = genjax.normal.assess(v, custom_tree.x, custom_tree.y)
        return w

    def random_weighted(self, key, custom_tree):
        return genjax.normal.random_weighted(key, custom_tree.x, custom_tree.y)


CustomNormal = _CustomNormal()


@genjax.static_gen_fn
def custom_normal(custom_tree):
    y = CustomNormal(custom_tree) @ "y"
    return CustomTree(y, y)


class TestStaticGenFnCustomPytree:
    def test_simple_normal_simulate(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        fn = jax.jit(simple_normal.simulate)
        tr = fn(key, (init_tree,))
        choice = tr.get_sample()
        (_, score1, _) = genjax.normal.importance(
            key, choice.get_submap("y1"), (init_tree.x, 1.0)
        )
        (_, score2, _) = genjax.normal.importance(
            key, choice.get_submap("y2"), (init_tree.y, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_custom_normal_simulate(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        fn = jax.jit(custom_normal.simulate)
        tr = fn(key, (init_tree,))
        choice = tr.get_sample()
        (_, score, _) = genjax.normal.importance(
            key, choice.get_submap("y"), (init_tree.x, init_tree.y)
        )
        test_score = score
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_importance(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        choice = C.n.at["y1"].set(5.0)
        fn = jax.jit(simple_normal.importance)
        (tr, w, _) = fn(key, choice, (init_tree,))
        choice = tr.get_sample()
        (_, score1, _) = genjax.normal.importance(
            key, choice.get_submap("y1"), (init_tree.x, 1.0)
        )
        (_, score2, _) = genjax.normal.importance(
            key, choice.get_submap("y2"), (init_tree.y, 1.0)
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)
        assert w == pytest.approx(score1, 0.01)


class TestStaticGenFnGradients:
    def test_simple_normal_assess(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        tr = jax.jit(simple_normal.simulate)(key, ())
        jitted = jax.jit(simple_normal.assess)
        choice = tr.get_sample()
        (score, _) = jitted(choice, ())
        assert score == tr.get_score()


class TestStaticGenFnImportance:
    def test_importance_simple_normal(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        fn = simple_normal.importance
        choice = C.n.at["y1"].set(0.5).at["y2"].set(0.5)
        key, sub_key = jax.random.split(key)
        (out, _, _) = fn(sub_key, choice, ())
        (_, score_1, _) = genjax.normal.importance(
            key, choice.get_submap("y1"), (0.0, 1.0)
        )
        (_, score_2, _) = genjax.normal.importance(
            key, choice.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score_1 + score_2
        assert choice["y1"] == out.get_sample()["y1"]
        assert choice["y2"] == out.get_sample()["y2"]
        assert out.get_score() == pytest.approx(test_score, 0.01)

    def test_importance_weight_correctness(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        # Full constraints.
        key = jax.random.PRNGKey(314159)
        choice = C.n.at["y1"].set(0.5).at["y2"].set(0.5)
        (tr, w, _) = simple_normal.importance(key, choice, ())
        y1 = tr.get_sample()["y1"]
        y2 = tr.get_sample()["y2"]
        assert y1 == 0.5
        assert y2 == 0.5
        (_, score_1, _) = genjax.normal.importance(
            key, choice.get_submap("y1"), (0.0, 1.0)
        )
        (_, score_2, _) = genjax.normal.importance(
            key, choice.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(test_score, 0.0001)

        # Partial constraints.
        choice = C.n.at["y2"].set(0.5)
        (tr, w, _) = simple_normal.importance(key, choice, ())
        tr_chm = tr.get_sample()
        y1 = tr_chm.get_submap("y1")
        y2 = tr_chm.get_submap("y2")
        assert tr_chm["y2"] == 0.5
        score_1, _ = genjax.normal.assess(y1, (0.0, 1.0))
        score_2, _ = genjax.normal.assess(y2, (0.0, 1.0))
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(score_2, 0.0001)

        # No constraints.
        choice = C.n
        (tr, w, _) = simple_normal.importance(key, choice, ())
        tr_chm = tr.get_sample()
        y1 = tr_chm.get_submap("y1")
        y2 = tr_chm.get_submap("y2")
        score_1, _ = genjax.normal.assess(y1, (0.0, 1.0))
        score_2, _ = genjax.normal.assess(y2, (0.0, 1.0))
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == 0.0


####################################################
#          Remember: the update weight math        #
#                                                  #
#   log p(r′,t′;x′) + log q(r;x,t) - log p(r,t;x)  #
#       - log q(r′;x′,t′) - q(t′;x′,t+u)           #
#                                                  #
####################################################


class TestStaticGenFnUpdate:
    def test_simple_normal_update(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_normal.update)

        new = C.n.at["y1"].set(2.0)
        original_choice = tr.get_sample()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = jitted(sub_key, tr, new, ())
        updated_choice = updated.get_sample()
        _y1 = updated_choice["y1"]
        _y2 = updated_choice["y2"]
        (_, score1, _) = genjax.normal.importance(
            key, updated_choice.get_submap("y1"), (0.0, 1.0)
        )
        (_, score2, _) = genjax.normal.importance(
            key, updated_choice.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert original_choice[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

        new = C.n.at["y1"].set(2.0).at["y2"].set(3.0)
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = jitted(sub_key, tr, new, ())
        updated_choice = updated.get_sample()
        _y1 = updated_choice.get_submap("y1")
        _y2 = updated_choice.get_submap("y2")
        (_, score1, _) = genjax.normal.importance(key, _y1, (0.0, 1.0))
        (_, score2, _) = genjax.normal.importance(key, _y2, (0.0, 1.0))
        test_score = score1 + score2
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_linked_normal_update(self):
        @genjax.static_gen_fn
        def simple_linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            y3 = genjax.normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_linked_normal.update)

        new = C.n.at["y1"].set(2.0)
        original_choice = tr.get_sample()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = jitted(sub_key, tr, new, ())
        updated_choice = updated.get_sample()
        y1 = updated_choice["y1"]
        y2 = updated_choice["y2"]
        y3 = updated_choice.get_submap("y3")
        score1, _ = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        score2, _ = genjax.normal.assess(C.v(y2), (y1, 1.0))
        score3, _ = genjax.normal.assess(y3, (y1 + y2, 1.0))
        test_score = score1 + score2 + score3
        assert original_choice["y1"] == discard["y1"]
        assert updated.get_score() == pytest.approx(original_score + w, 0.01)
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_hierarchical_normal(self):
        @genjax.static_gen_fn
        def _inner(x):
            y1 = genjax.normal(x, 1.0) @ "y1"
            return y1

        @genjax.static_gen_fn
        def simple_hierarchical_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = _inner(y1) @ "y2"
            y3 = _inner(y1 + y2) @ "y3"
            return y1 + y2 + y3

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_hierarchical_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_hierarchical_normal.update)

        new = C.n.at["y1"].set(2.0)
        original_choice = tr.get_sample()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = jitted(sub_key, tr, new, ())
        updated_choice = updated.get_sample()
        y1 = updated_choice["y1"]
        y2 = updated_choice["y2", "y1"]
        y3 = updated_choice["y3", "y1"]
        assert y1 == new["y1"]
        assert y2 == original_choice["y2", "y1"]
        assert y3 == original_choice["y3", "y1"]
        score1, _ = genjax.normal.assess(C.v(y1), (0.0, 1.0))
        score2, _ = genjax.normal.assess(C.v(y2), (y1, 1.0))
        score3, _ = genjax.normal.assess(C.v(y3), (y1 + y2, 1.0))
        test_score = score1 + score2 + score3
        assert original_choice["y1"] == discard["y1"]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_update_weight_correctness(self):
        @genjax.static_gen_fn
        def simple_linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            y3 = genjax.normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal.simulate)(sub_key, ())
        jitted = jax.jit(simple_linked_normal.update)

        old_y1 = tr.get_sample()["y1"]
        old_y2 = tr.get_sample()["y2"]
        old_y3 = tr.get_sample()["y3"]
        new_y1 = 2.0
        new = C.n.at["y1"].set(new_y1)
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = jitted(sub_key, tr, new, ())

        # TestStaticGenFn new scores.
        updated_sample = updated.get_sample()
        assert updated_sample["y1"] == new_y1
        sel = genjax.Selection.at["y1"]
        update_spec = RemoveSelectionUpdateSpec(sel)
        assert (
            updated.project(key, update_spec)
            == genjax.normal.assess(C.v(new_y1), (0.0, 1.0))[0]
        )
        assert updated_sample["y2"] == old_y2
        sel = genjax.Selection.at["y2"]
        update_spec = RemoveSelectionUpdateSpec(sel)
        assert updated.project(key, update_spec) == pytest.approx(
            genjax.normal.assess(C.v(old_y2), (new_y1, 1.0))[0], 0.0001
        )
        assert updated.get_sample()["y3"] == old_y3
        sel = genjax.Selection.at["y3"]
        update_spec = RemoveSelectionUpdateSpec(sel)
        assert updated.project(key, update_spec) == pytest.approx(
            genjax.normal.assess(C.v(old_y3), (new_y1 + old_y2, 1.0))[0], 0.0001
        )

        # TestStaticGenFn weight correctness.
        δ_y3 = (
            genjax.normal.assess(C.v(old_y3), (new_y1 + old_y2, 1.0))[0]
            - genjax.normal.assess(C.v(old_y3), (old_y1 + old_y2, 1.0))[0]
        )
        δ_y2 = (
            genjax.normal.assess(C.v(old_y2), (new_y1, 1.0))[0]
            - genjax.normal.assess(C.v(old_y2), (old_y1, 1.0))[0]
        )
        δ_y1 = (
            genjax.normal.assess(C.v(new_y1), (0.0, 1.0))[0]
            - genjax.normal.assess(C.v(old_y1), (0.0, 1.0))[0]
        )
        assert w == pytest.approx((δ_y3 + δ_y2 + δ_y1), 0.0001)

        # TestStaticGenFn composition of update calls.
        new_y3 = 2.0
        new = C.n.at["y3"].set(new_y3)
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = jitted(sub_key, updated, new, ())
        assert updated.get_sample()["y3"] == 2.0
        correct_w = (
            genjax.normal.assess(C.v(new_y3), (new_y1 + old_y2, 1.0))[0]
            - genjax.normal.assess(C.v(old_y3), (new_y1 + old_y2, 1.0))[0]
        )
        assert w == pytest.approx(correct_w, 0.0001)

    def test_update_pytree_argument(self):
        @Pytree.dataclass
        class SomePytree(genjax.Pytree):
            x: FloatArray
            y: FloatArray

        @genjax.static_gen_fn
        def simple_linked_normal_with_tree_argument(tree):
            y1 = genjax.normal(tree.x, tree.y) @ "y1"
            return y1

        key = jax.random.PRNGKey(314159)
        init_tree = SomePytree(0.0, 1.0)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal_with_tree_argument.simulate)(
            sub_key, (init_tree,)
        )
        jitted = jax.jit(simple_linked_normal_with_tree_argument.update)
        new_y1 = 2.0
        constraints = C.n.at["y1"].set(new_y1)
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = jitted(
            sub_key, tr, constraints, (Diff.tree_diff_no_change(init_tree),)
        )
        assert updated.get_sample()["y1"] == new_y1
        new_tree = SomePytree(1.0, 2.0)
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = jitted(
            sub_key, tr, constraints, (Diff.tree_diff_unknown_change(new_tree),)
        )
        assert updated.get_sample()["y1"] == new_y1


#####################
# Language features #
#####################


class TestStaticGenFnStaticAddressChecks:
    def test_simple_normal_addr_dup(self):
        @genjax.static_gen_fn
        def simple_normal_addr_dup():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y1"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(AddressReuse) as exc_info:
            _ = simple_normal_addr_dup.simulate(key, ())
        assert exc_info.value.args == ("y1",)

    def test_simple_normal_addr_tracer(self):
        @genjax.static_gen_fn
        def simple_normal_addr_tracer():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ y1
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(StaticAddressJAX) as exc_info:
            _ = simple_normal_addr_tracer.simulate(key, ())
        assert isinstance(exc_info.value.args[0], DynamicJaxprTracer)


class TestStaticGenFnForwardRef:
    def test_forward_ref(self):
        def make_gen_fn():
            @genjax.static_gen_fn
            def proposal(x):
                x = outlier(x) @ "x"
                return x

            @genjax.static_gen_fn
            def outlier(prob):
                is_outlier = genjax.bernoulli(prob) @ "is_outlier"
                return is_outlier

            return proposal

        key = jax.random.PRNGKey(314159)
        proposal = make_gen_fn()
        _tr = proposal.simulate(key, (0.3,))
        assert True


class TestStaticGenFnInline:
    def test_inline_simulate(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.static_gen_fn
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.static_gen_fn
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(higher_model.simulate)(sub_key, ())
        choices = tr.get_sample()
        assert choices.has_submap("y1")
        assert choices.has_submap("y2")
        tr = jax.jit(higher_higher_model.simulate)(key, ())
        choices = tr.get_sample()
        assert choices.has_submap("y1")
        assert choices.has_submap("y2")

    def test_inline_importance(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.static_gen_fn
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.static_gen_fn
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        choice = C.n.at["y1"].set(3.0)
        key, sub_key = jax.random.split(key)
        (tr, w, _) = jax.jit(higher_model.importance)(sub_key, choice, ())
        choices = tr.get_sample()
        assert w == genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]
        (tr, w, _) = jax.jit(higher_higher_model.importance)(key, choice, ())
        choices = tr.get_sample()
        assert w == genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]

    def test_inline_update(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.static_gen_fn
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.static_gen_fn
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        choice = C.n.at["y1"].set(3.0)
        tr = jax.jit(higher_model.simulate)(sub_key, ())
        old_value = tr.get_sample().get_submap("y1")
        key, sub_key = jax.random.split(key)
        (tr, w, rd, _) = jax.jit(higher_model.update)(sub_key, tr, choice, ())
        choices = tr.get_sample()
        assert (
            w
            == genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]
            - genjax.normal.assess(old_value, (0.0, 1.0))[0]
        )
        key, sub_key = jax.random.split(key)
        tr = jax.jit(higher_higher_model.simulate)(sub_key, ())
        old_value = tr.get_sample().get_submap("y1")
        (tr, w, rd, _) = jax.jit(higher_higher_model.update)(key, tr, choice, ())
        choices = tr.get_sample()
        assert w == pytest.approx(
            genjax.normal.assess(choices.get_submap("y1"), (0.0, 1.0))[0]
            - genjax.normal.assess(old_value, (0.0, 1.0))[0],
            0.0001,
        )

    def test_inline_assess(self):
        @genjax.static_gen_fn
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @genjax.static_gen_fn
        def higher_model():
            y = simple_normal.inline()
            return y

        @genjax.static_gen_fn
        def higher_higher_model():
            y = higher_model.inline()
            return y

        _key = jax.random.PRNGKey(314159)
        choice = C.n.at["y1"].set(3.0).at["y2"].set(3.0)
        (score, ret) = jax.jit(higher_model.assess)(choice, ())
        assert (
            score
            == genjax.normal.assess(choice.get_submap("y1"), (0.0, 1.0))[0]
            + genjax.normal.assess(choice.get_submap("y2"), (0.0, 1.0))[0]
        )
        (score, ret) = jax.jit(higher_higher_model.assess)(choice, ())
        assert (
            score
            == genjax.normal.assess(choice.get_submap("y1"), (0.0, 1.0))[0]
            + genjax.normal.assess(choice.get_submap("y2"), (0.0, 1.0))[0]
        )
