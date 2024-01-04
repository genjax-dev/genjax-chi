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

from dataclasses import dataclass
from typing import Any

import genjax
import jax
import jax.numpy as jnp
import pytest
from genjax import ExactDensity, FloatArray
from genjax._src.core.interpreters.incremental import (
    tree_diff_no_change,
    tree_diff_unknown_change,
)
from genjax._src.generative_functions.interpreted import trace
from jaxtyping import ArrayLike

#############
# Datatypes #
#############

##################################
# Generative function interfaces #
##################################


with_both_languages = pytest.mark.parametrize(
    "lang", (genjax.Interpreted, genjax.Static), ids=("Interpreted", "Static")
)


@with_both_languages
class TestSimulate:
    def test_simple_normal_sugar(self, lang):
        @lang
        def normal_sugar():
            y = genjax.normal(0.0, 1.0) @ "y"
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = normal_sugar.simulate(sub_key, ())
        chm = tr.get_choices()
        _, score = genjax.normal.importance(key, chm.get_submap("y"), (0.0, 1.0))
        assert tr.get_score() == pytest.approx(score, 0.01)

    def test_simple_normal_simulate(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())
        chm = tr.get_choices()
        (_, score1) = genjax.normal.importance(key, chm.get_submap("y1"), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, chm.get_submap("y2"), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_multiple_returns(self, lang):
        @lang
        def simple_normal_multiple_returns():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal_multiple_returns.simulate(sub_key, ())
        y1_ = tr["y1"]
        y2_ = tr["y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (_, score1) = genjax.normal.importance(key, genjax.choice_value(y1), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, genjax.choice_value(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_hierarchical_simple_normal_multiple_returns(self, lang):
        @lang
        def _submodel():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1, y2

        @lang
        def hierarchical_simple_normal_multiple_returns():
            y1, y2 = _submodel() @ "y1"
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = hierarchical_simple_normal_multiple_returns.simulate(sub_key, ())
        y1_ = tr["y1", "y1"]
        y2_ = tr["y1", "y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_
        assert y2 == y2_
        (_, score1) = genjax.normal.importance(key, genjax.choice_value(y1), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, genjax.choice_value(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)


@with_both_languages
class TestAssess:
    def test_simple_normal_assess(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())
        chm = tr.get_choices().strip()
        (score, retval) = simple_normal.assess(chm, ())
        assert score == tr.get_score()


@with_both_languages
class TestClosureConvert:
    def test_closure_convert(self, lang):
        def emits_cc_gen_fn(v):
            @lang
            @genjax.dynamic_closure(v)
            def model(v):
                x = genjax.normal(jnp.sum(v), 1.0) @ "x"
                return x

            return model

        @lang
        def model():
            x = jnp.ones(5)
            gen_fn = emits_cc_gen_fn(x)
            v = gen_fn() @ "x"
            return (v, gen_fn)

        key = jax.random.PRNGKey(314159)
        _ = model.simulate(key, ())
        assert True


@dataclass
class CustomTree(genjax.Pytree):
    x: Any
    y: Any

    def flatten(self):
        return (self.x, self.y), ()


@genjax.Interpreted
def simple_normal(custom_tree):
    y1 = trace("y1", genjax.normal)(custom_tree.x, 1.0)
    y2 = trace("y2", genjax.normal)(custom_tree.y, 1.0)
    return CustomTree(y1, y2)


@dataclass
class _CustomNormal(ExactDensity):
    def logpdf(self, v, custom_tree):
        return genjax.normal.logpdf(v, custom_tree.x, custom_tree.y)

    def sample(self, key, custom_tree):
        return genjax.normal.sample(key, custom_tree.x, custom_tree.y)


CustomNormal = _CustomNormal()


@genjax.Interpreted
def custom_normal(custom_tree):
    y = CustomNormal(custom_tree) @ "y"
    return CustomTree(y, y)


class TestCustomPytree:
    def test_simple_normal_simulate(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        tr = simple_normal.simulate(key, (init_tree,))
        chm = tr.get_choices()
        (_, score1) = genjax.normal.importance(
            key,
            chm.get_submap("y1"),
            (init_tree.x, 1.0),
        )
        (_, score2) = genjax.normal.importance(
            key,
            chm.get_submap("y2"),
            (init_tree.y, 1.0),
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_custom_normal_simulate(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        tr = custom_normal.simulate(key, (init_tree,))
        chm = tr.get_choices()
        (_, score) = genjax.normal.importance(
            key, chm.get_submap("y"), (init_tree.x, init_tree.y)
        )
        test_score = score
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_importance(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        chm = genjax.choice_map({"y1": 5.0})
        (tr, w) = simple_normal.importance(key, chm, (init_tree,))
        chm = tr.get_choices()
        (_, score1) = genjax.normal.importance(
            key,
            chm.get_submap("y1"),
            (init_tree.x, 1.0),
        )
        (_, score2) = genjax.normal.importance(
            key,
            chm.get_submap("y2"),
            (init_tree.y, 1.0),
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)
        assert w == pytest.approx(score1, 0.01)


@with_both_languages
class TestGradients:
    def test_simple_normal_assess(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        tr = simple_normal.simulate(key, ())
        chm = tr.get_choices()
        (score, _) = simple_normal.assess(chm, ())
        assert score == tr.get_score()


@with_both_languages
class TestImportance:
    def test_importance_simple_normal(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        fn = simple_normal.importance
        chm = genjax.choice_map({("y1",): 0.5, ("y2",): 0.5})
        key, sub_key = jax.random.split(key)
        (tr, _) = fn(sub_key, chm, ())
        out = tr.get_choices()
        y1 = chm[("y1",)]
        y2 = chm[("y2",)]
        (_, score_1) = genjax.normal.importance(key, chm.get_submap("y1"), (0.0, 1.0))
        (_, score_2) = genjax.normal.importance(key, chm.get_submap("y2"), (0.0, 1.0))
        test_score = score_1 + score_2
        assert y1 == out[("y1",)]
        assert y2 == out[("y2",)]
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_importance_weight_correctness(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        # Full constraints.
        key = jax.random.PRNGKey(314159)
        chm = genjax.choice_map({("y1",): 0.5, ("y2",): 0.5})
        (tr, w) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"]
        y2 = tr["y2"]
        assert y1 == 0.5
        assert y2 == 0.5
        (_, score_1) = genjax.normal.importance(key, chm.get_submap("y1"), (0.0, 1.0))
        (_, score_2) = genjax.normal.importance(key, chm.get_submap("y2"), (0.0, 1.0))
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(test_score, 0.0001)

        # Partial constraints.
        chm = genjax.choice_map({("y2",): 0.5})
        (tr, w) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"]
        y2 = tr["y2"]
        assert y2 == 0.5
        score_1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score_2 = genjax.normal.logpdf(y2, 0.0, 1.0)
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(score_2, 0.0001)

        # No constraints.
        # chm = genjax.EmptyChoice()
        # NB: in the interpreted language, get_submap will be called on this item,
        # so beartype thinks this object should have that capability. Oddly,
        # genjax.ChoiceMap() is not the same animal as genjax.choice_map({}). The
        # former doesn't work. This needs cleaning up.
        chm = genjax.choice_map({})
        (tr, w) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"]
        y2 = tr["y2"]
        score_1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score_2 = genjax.normal.logpdf(y2, 0.0, 1.0)
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


@with_both_languages
class TestUpdate:
    def test_simple_normal_update(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())

        new = genjax.choice_map({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = simple_normal.update(sub_key, tr, new, ())
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        (_, score1) = genjax.normal.importance(
            key, updated_chm.get_submap("y1"), (0.0, 1.0)
        )
        (_, score2) = genjax.normal.importance(
            key, updated_chm.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

        new = genjax.choice_map({("y1",): 2.0, ("y2",): 3.0})
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = simple_normal.update(sub_key, tr, new, ())
        updated_chm = updated.get_choices()
        y1 = updated_chm[("y1",)]
        y2 = updated_chm[("y2",)]
        (_, score1) = genjax.normal.importance(
            key, updated_chm.get_submap("y1"), (0.0, 1.0)
        )
        (_, score2) = genjax.normal.importance(
            key, updated_chm.get_submap("y2"), (0.0, 1.0)
        )
        test_score = score1 + score2
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_linked_normal_update(self, lang):
        @lang
        def simple_linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            y3 = genjax.normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_linked_normal.simulate(sub_key, ())

        new = genjax.choice_map({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = simple_linked_normal.update(sub_key, tr, new, ())
        updated_chm = updated.get_choices().strip()
        y1 = updated_chm["y1"]
        y2 = updated_chm["y2"]
        y3 = updated_chm["y3"]
        score1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score2 = genjax.normal.logpdf(y2, y1, 1.0)
        score3 = genjax.normal.logpdf(y3, y1 + y2, 1.0)
        test_score = score1 + score2 + score3
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == pytest.approx(original_score + w, 0.01)
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_hierarchical_normal(self, lang):
        @lang
        def _inner(x):
            y1 = genjax.normal(x, 1.0) @ "y1"
            return y1

        @lang
        def simple_hierarchical_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = _inner(y1) @ "y2"
            y3 = _inner(y1 + y2) @ "y3"
            return y1 + y2 + y3

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_hierarchical_normal.simulate(sub_key, ())

        new = genjax.choice_map({("y1",): 2.0})
        original_chm = tr.get_choices()
        original_score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (updated, w, _, discard) = simple_hierarchical_normal.update(
            sub_key, tr, new, ()
        )
        updated_chm = updated.get_choices().strip()
        y1 = updated_chm["y1"]
        y2 = updated_chm["y2", "y1"]
        y3 = updated_chm["y3", "y1"]
        assert y1 == new["y1"]
        assert y2 == original_chm["y2", "y1"]
        assert y3 == original_chm["y3", "y1"]
        score1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score2 = genjax.normal.logpdf(y2, y1, 1.0)
        score3 = genjax.normal.logpdf(y3, y1 + y2, 1.0)
        test_score = score1 + score2 + score3
        assert original_chm[("y1",)] == discard[("y1",)]
        assert updated.get_score() == original_score + w
        assert updated.get_score() == pytest.approx(test_score, 0.01)

    def test_update_weight_correctness(self, lang):
        @lang
        def simple_linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(y1, 1.0) @ "y2"
            y3 = genjax.normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_linked_normal.simulate(sub_key, ())

        old_y1 = tr["y1"]
        old_y2 = tr["y2"]
        old_y3 = tr["y3"]
        new_y1 = 2.0
        new = genjax.choice_map({("y1",): new_y1})
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = simple_linked_normal.update(sub_key, tr, new, ())

        # Test new scores.
        assert updated["y1"] == new_y1
        sel = genjax.select("y1")
        assert updated.project(sel) == genjax.normal.logpdf(new_y1, 0.0, 1.0)
        assert updated["y2"] == old_y2
        sel = genjax.select("y2")
        assert updated.project(sel) == pytest.approx(
            genjax.normal.logpdf(old_y2, new_y1, 1.0), 0.0001
        )
        assert updated["y3"] == old_y3
        sel = genjax.select("y3")
        assert updated.project(sel) == pytest.approx(
            genjax.normal.logpdf(old_y3, new_y1 + old_y2, 1.0), 0.0001
        )

        # Test weight correctness.
        δ_y3 = genjax.normal.logpdf(
            old_y3, new_y1 + old_y2, 1.0
        ) - genjax.normal.logpdf(old_y3, old_y1 + old_y2, 1.0)
        δ_y2 = genjax.normal.logpdf(old_y2, new_y1, 1.0) - genjax.normal.logpdf(
            old_y2, old_y1, 1.0
        )
        δ_y1 = genjax.normal.logpdf(new_y1, 0.0, 1.0) - genjax.normal.logpdf(
            old_y1, 0.0, 1.0
        )
        assert w == pytest.approx((δ_y3 + δ_y2 + δ_y1), 0.0001)

        # Test composition of update calls.
        new_y3 = 2.0
        new = genjax.choice_map({("y3",): new_y3})
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = simple_linked_normal.update(sub_key, updated, new, ())
        assert updated["y3"] == 2.0
        correct_w = genjax.normal.logpdf(
            new_y3, new_y1 + old_y2, 1.0
        ) - genjax.normal.logpdf(old_y3, new_y1 + old_y2, 1.0)
        assert w == pytest.approx(correct_w, 0.0001)

    def test_update_pytree_argument(self, lang):
        @dataclass
        class SomePytree(genjax.Pytree):
            x: ArrayLike
            y: ArrayLike

            def flatten(self):
                return (self.x, self.y), ()

        @lang
        def simple_linked_normal_with_tree_argument(tree):
            y1 = genjax.normal(tree.x, tree.y) @ "y1"
            return y1

        key = jax.random.PRNGKey(314159)
        init_tree = SomePytree(0.0, 1.0)
        key, sub_key = jax.random.split(key)
        tr = simple_linked_normal_with_tree_argument.simulate(sub_key, (init_tree,))
        new_y1 = 2.0
        constraints = genjax.choice_map({("y1",): new_y1})
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = simple_linked_normal_with_tree_argument.update(
            sub_key, tr, constraints, (tree_diff_no_change(init_tree),)
        )
        assert updated["y1"] == new_y1
        new_tree = SomePytree(1.0, 2.0)
        key, sub_key = jax.random.split(key)
        (updated, w, _, _) = simple_linked_normal_with_tree_argument.update(
            sub_key, tr, constraints, (tree_diff_unknown_change(new_tree),)
        )
        assert updated["y1"] == new_y1


#####################
# Language features #
#####################


@with_both_languages
class TestInterpretedLanguageSugar:
    def test_interpreted_sugar(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        tr = simple_normal.simulate(key, ())

        key = jax.random.PRNGKey(314159)
        v = simple_normal(key, ())
        assert tr.get_retval() == v


@with_both_languages
class TestInterpretedAddressChecks:
    def test_simple_normal_addr_dup(self, lang):
        @lang
        def simple_normal_addr_dup():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y1"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(Exception):
            _ = simple_normal_addr_dup.simulate(key, ())

    def test_simple_normal_addr_tracer(self, lang):
        @lang
        def simple_normal_addr_tracer():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ y1
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(Exception):
            _ = genjax.simulate(simple_normal_addr_tracer)(key, ())


@with_both_languages
class TestForwardRef:
    def test_forward_ref(self, lang):
        def make_gen_fn():
            @lang
            def proposal(x):
                x = outlier(x) @ "x"
                return x

            @lang
            def outlier(prob):
                is_outlier = genjax.bernoulli(prob) @ "is_outlier"
                return is_outlier

            return proposal

        key = jax.random.PRNGKey(314159)
        proposal = make_gen_fn()
        proposal.simulate(key, (0.3,))
        assert True


@with_both_languages
class TestInline:
    def test_inline_simulate(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @lang
        def higher_model():
            y = simple_normal.inline()
            return y

        @lang
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = higher_model.simulate(sub_key, ())
        choices = tr.strip()
        assert choices.has_submap("y1")
        assert choices.has_submap("y2")
        tr = higher_higher_model.simulate(key, ())
        choices = tr.strip()
        assert choices.has_submap("y1")
        assert choices.has_submap("y2")

    def test_inline_importance(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @lang
        def higher_model():
            y = simple_normal.inline()
            return y

        @lang
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        chm = genjax.choice_map({"y1": 3.0})
        key, sub_key = jax.random.split(key)
        (tr, w) = higher_model.importance(sub_key, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(choices["y1"], 0.0, 1.0)
        (tr, w) = higher_higher_model.importance(key, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(choices["y1"], 0.0, 1.0)

    def test_inline_update(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @lang
        def higher_model():
            y = simple_normal.inline()
            return y

        @lang
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        chm = genjax.choice_map({"y1": 3.0})
        tr = higher_model.simulate(sub_key, ())
        old_value = tr.strip()["y1"]
        key, sub_key = jax.random.split(key)
        (tr, w, rd, _) = higher_model.update(sub_key, tr, chm, ())
        choices = tr.strip()
        assert w == genjax.normal.logpdf(
            choices["y1"], 0.0, 1.0
        ) - genjax.normal.logpdf(old_value, 0.0, 1.0)
        key, sub_key = jax.random.split(key)
        tr = higher_higher_model.simulate(sub_key, ())
        old_value = tr.strip()["y1"]
        (tr, w, rd, _) = higher_higher_model.update(key, tr, chm, ())
        choices = tr.strip()
        assert w == pytest.approx(
            genjax.normal.logpdf(choices["y1"], 0.0, 1.0)
            - genjax.normal.logpdf(old_value, 0.0, 1.0),
            0.0001,
        )

    def test_inline_assess(self, lang):
        @lang
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        @lang
        def higher_model():
            y = simple_normal.inline()
            return y

        @lang
        def higher_higher_model():
            y = higher_model.inline()
            return y

        key = jax.random.PRNGKey(314159)
        chm = genjax.choice_map({"y1": 3.0, "y2": 3.0})
        (score, ret) = higher_model.assess(chm, ())
        assert score == genjax.normal.logpdf(
            chm["y1"], 0.0, 1.0
        ) + genjax.normal.logpdf(chm["y2"], 0.0, 1.0)
        (score, ret) = higher_higher_model.assess(chm, ())
        assert score == genjax.normal.logpdf(
            chm["y1"], 0.0, 1.0
        ) + genjax.normal.logpdf(chm["y2"], 0.0, 1.0)
