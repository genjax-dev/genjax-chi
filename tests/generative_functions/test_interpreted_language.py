import jax
import genjax
import pytest
from genjax._src.generative_functions.interpreted import trace


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
