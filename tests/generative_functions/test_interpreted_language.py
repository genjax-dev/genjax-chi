from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
import genjax
import pytest

from genjax import ExactDensity
from genjax._src.generative_functions.interpreted import (
    trace,
    InterpretedGenerativeFunction,
)


class TestSimulate:
    def test_simple_normal_sugar(self):
        @genjax.lang(genjax.Interpreted)
        def normal_sugar():
            y = genjax.normal(0.0, 1.0) @ "y"
            return y

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = normal_sugar.simulate(sub_key, ())
        chm = tr.get_choices()
        _, score = genjax.normal.importance(key, chm["y"].get_choices(), (0.0, 1.0))
        assert tr.get_score() == pytest.approx(score, 0.01)

    def test_simple_normal_simulate(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal():
            y1 = trace("y1", genjax.normal)(0.0, 1.0)
            y2 = trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())
        chm = tr.get_choices()
        (_, score1) = genjax.normal.importance(key, chm["y1"].get_choices(), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, chm["y2"].get_choices(), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_simple_normal_multiple_returns(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal_multiple_returns():
            y1 = trace("y1", genjax.normal)(0.0, 1.0)
            y2 = trace("y2", genjax.normal)(0.0, 1.0)
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal_multiple_returns.simulate(sub_key, ())
        y1_ = tr["y1"]
        y2_ = tr["y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_.get_value()
        assert y2 == y2_.get_value()
        (_, score1) = genjax.normal.importance(key, genjax.choice_value(y1), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, genjax.choice_value(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_hierarchical_simple_normal_multiple_returns(self):
        @genjax.lang(genjax.Interpreted)
        def _submodel():
            y1 = trace("y1", genjax.normal)(0.0, 1.0)
            y2 = trace("y2", genjax.normal)(0.0, 1.0)
            return y1, y2

        @genjax.lang(genjax.Interpreted)
        def hierarchical_simple_normal_multiple_returns():
            y1, y2 = trace("y1", _submodel)()
            return y1, y2

        key = jax.random.PRNGKey(314159)
        key, sub_key = jax.random.split(key)
        tr = hierarchical_simple_normal_multiple_returns.simulate(sub_key, ())
        y1_ = tr["y1", "y1"]
        y2_ = tr["y1", "y2"]
        y1, y2 = tr.get_retval()
        assert y1 == y1_.get_value()
        assert y2 == y2_.get_value()
        (_, score1) = genjax.normal.importance(key, genjax.choice_value(y1), (0.0, 1.0))
        (_, score2) = genjax.normal.importance(key, genjax.choice_value(y2), (0.0, 1.0))
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)


class TestAssess:
    def test_simple_normal_assess(self):
        @genjax.lang(genjax.Interpreted)
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


class TestClosureConvert:
    def test_closure_convert(self):
        def emits_cc_gen_fn(v):
            @genjax.lang(genjax.Interpreted)
            @genjax.dynamic_closure(v)
            def model(v):
                x = genjax.normal(jnp.sum(v), 1.0) @ "x"
                return x

            return model

        @genjax.lang(genjax.Interpreted)
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


@genjax.lang(genjax.Interpreted)
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


@genjax.lang(genjax.Interpreted)
def custom_normal(custom_tree):
    y = trace("y", CustomNormal)(custom_tree)
    return CustomTree(y, y)


class TestCustomPytree:
    def test_simple_normal_simulate(self):
        key = jax.random.PRNGKey(314159)
        init_tree = CustomTree(3.0, 5.0)
        tr = simple_normal.simulate(key, (init_tree,))
        chm = tr.get_choices()
        (_, score1) = genjax.normal.importance(
            # TODO: note for McCoy: in Static, .get_choices() is not needed here
            key,
            chm.get_submap("y1").get_choices(),
            (init_tree.x, 1.0),
        )
        (_, score2) = genjax.normal.importance(
            # TODO: note for McCoy: in Static, .get_choices() is not needed here
            key,
            chm.get_submap("y2").get_choices(),
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
            key, chm.get_submap("y").get_choices(), (init_tree.x, init_tree.y)
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
            # TODO: note for McCoy: in Static, .get_choices() is not needed here
            key,
            chm.get_submap("y1").get_choices(),
            (init_tree.x, 1.0),
        )
        (_, score2) = genjax.normal.importance(
            # TODO: note for McCoy: in Static, .get_choices() is not needed here
            key,
            chm.get_submap("y2").get_choices(),
            (init_tree.y, 1.0),
        )
        test_score = score1 + score2
        assert tr.get_score() == pytest.approx(test_score, 0.01)
        assert w == pytest.approx(score1, 0.01)


class TestGradients:
    def test_simple_normal_assess(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal():
            y1 = genjax.trace("y1", genjax.normal)(0.0, 1.0)
            y2 = genjax.trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        tr = simple_normal.simulate(key, ())
        chm = tr.get_choices()
        (score, _) = simple_normal.assess(chm, ())
        assert score == tr.get_score()


class TestImportance:
    def test_importance_simple_normal(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal():
            y1 = trace("y1", genjax.normal)(0.0, 1.0)
            y2 = trace("y2", genjax.normal)(0.0, 1.0)
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
        # TODO: get_value() not needed in Static language
        assert y1 == out[("y1",)].get_value()
        assert y2 == out[("y2",)].get_value()
        assert tr.get_score() == pytest.approx(test_score, 0.01)

    def test_importance_weight_correctness(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal():
            y1 = trace("y1", genjax.normal)(0.0, 1.0)
            y2 = trace("y2", genjax.normal)(0.0, 1.0)
            return y1 + y2

        # Full constraints.
        key = jax.random.PRNGKey(314159)
        chm = genjax.choice_map({("y1",): 0.5, ("y2",): 0.5})
        (tr, w) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"]
        y2 = tr["y2"]
        # TODO: Static language does not require get_value here
        assert y1.get_value() == 0.5
        assert y2.get_value() == 0.5
        (_, score_1) = genjax.normal.importance(key, chm.get_submap("y1"), (0.0, 1.0))
        (_, score_2) = genjax.normal.importance(key, chm.get_submap("y2"), (0.0, 1.0))
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == pytest.approx(test_score, 0.0001)

        # Partial constraints.
        chm = genjax.choice_map({("y2",): 0.5})
        (tr, w) = simple_normal.importance(key, chm, ())
        y1 = tr["y1"].get_value()
        y2 = tr["y2"].get_value()
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
        # standard remark about get_value()
        y1 = tr["y1"].get_value()
        y2 = tr["y2"].get_value()
        score_1 = genjax.normal.logpdf(y1, 0.0, 1.0)
        score_2 = genjax.normal.logpdf(y2, 0.0, 1.0)
        test_score = score_1 + score_2
        assert tr.get_score() == pytest.approx(test_score, 0.0001)
        assert w == 0.0


#####################
# Language features #
#####################


class TestInterpretedLanguageSugar:
    def test_interpreted_sugar(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y2"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        tr = simple_normal.simulate(key, ())

        key = jax.random.PRNGKey(314159)
        v = simple_normal(key, ())
        assert tr.get_retval() == v


class TestInterpretedAddressChecks:
    def test_simple_normal_addr_dup(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal_addr_dup():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ "y1"
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(Exception):
            _ = genjax.simulate(simple_normal_addr_dup)(key, ())

    def test_simple_normal_addr_tracer(self):
        @genjax.lang(genjax.Interpreted)
        def simple_normal_addr_tracer():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            y2 = genjax.normal(0.0, 1.0) @ y1
            return y1 + y2

        key = jax.random.PRNGKey(314159)
        with pytest.raises(Exception):
            _ = genjax.simulate(simple_normal_addr_tracer)(key, ())


class TestForwardRef:
    def test_forward_ref(self):
        def make_gen_fn():
            @genjax.lang(genjax.Interpreted)
            def proposal(x):
                x = outlier(x) @ "x"
                return x

            @genjax.lang(genjax.Interpreted)
            def outlier(prob):
                is_outlier = genjax.bernoulli(prob) @ "is_outlier"
                return is_outlier

            return proposal

        key = jax.random.PRNGKey(314159)
        proposal = make_gen_fn()
        proposal.simulate(key, (0.3,))
        assert True
