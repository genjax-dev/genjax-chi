import jax
import jax.numpy as jnp
import genjax as gex
import pytest


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


def test_simple_normal_simulate(benchmark):
    key = jax.random.PRNGKey(314159)
    jitted = jax.jit(gex.simulate(simple_normal))
    tr = benchmark(jitted, key)
    chm = tr.get_choices()
    y1 = chm[("y1",)]
    y2 = chm[("y2",)]
    test_score = gex.Normal().score(y1) + gex.Normal().score(y2)
    assert tr.get_score() == pytest.approx(test_score, 0.01)
