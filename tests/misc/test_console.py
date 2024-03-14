import genjax
import jax.random


class TestConsole:
    K = jax.random.PRNGKey(314159)

    @genjax.unfold_combinator(max_length=100)
    @genjax.static_gen_fn
    def walk(prev, scale):
        dx = genjax.normal(0.0, scale) @ "dx"
        return prev + dx

    @genjax.static_gen_fn
    def model(n):
        loc = genjax.normal(10.0, 0.5) @ "loc"
        scale = genjax.beta(2.0, 5.0) @ "scale"
        steps = TestConsole.walk(n, loc, scale) @ "steps"
        return steps

    def test_console(self):
        c = genjax.console()
        c.print(0)

    def test_trace(self):
        _c = genjax.console()
        _tr = TestConsole.model.simulate(TestConsole.K, (5.0,))
        # TODO(colin): it's probably better to test this feature with
        # a Jupyter notebook, but we could create some small tests that
        # are worth doing here. We can decide that in code review
