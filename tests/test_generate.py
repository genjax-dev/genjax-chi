import jax
import jax.numpy as jnp
import genjax as gex
import pytest


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


def test_simple_normal_generate(benchmark):
    key = jax.random.PRNGKey(314159)
    jitted = jax.jit(gex.importance(simple_normal))
    chm = {("y1",): 0.5, ("y2",): 0.5}
    w, tr = benchmark(jitted, chm, key)
    out = tr.get_choices()
    y1 = chm[("y1",)]
    y2 = chm[("y2",)]
    test_score = gex.Normal().score(y1) + gex.Normal().score(y2)
    assert chm == out
    assert tr.get_score() == pytest.approx(test_score, 0.01)
