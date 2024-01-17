import genjax
import jax
import jax.numpy as jnp
from genjax import interpreted as gen


class TestBugHunt:
    @gen
    @staticmethod
    def simple_model(y, scale):
        """The simple_model is merely a gaussian distribution with mean `y` and
        standard deviation `scale`.
        """
        y = genjax.normal(y, scale) @ "value"
        return y

    @gen
    @staticmethod
    def compound_model(y, scale):
        """The compound model flips an unfair coin to decide whether to return
        an outlier (10% chance) or a regular value from simple_model. Outliers
        are drawn from a distribution with a much wider standard deviation.
        """
        is_outlier = genjax.flip(0.1) @ "outlier"
        scale = scale * 100 if is_outlier else scale
        return TestBugHunt.simple_model(y, scale) @ "y"

    def test_simple_model(self):
        # Fix the random number seed and sample a value from the distribution.

        key = jax.random.PRNGKey(0)
        tr = self.simple_model.simulate(key, (0.0, 1.0))
        assert jnp.abs(tr.get_retval() - jnp.array(-1.2515389)) < 1e-5

    def test_compound_model(self):
        # Fix the random number seed and sample a value from the distribution.
        key = jax.random.PRNGKey(0)
        tr = self.compound_model.simulate(key, (0.0, 1.0))

        choices = tr.get_choices()
        assert choices["outlier"] == jnp.array(0)
        assert jnp.abs(choices["y", "value"] - jnp.array(-0.16758952)) < 1e-5
