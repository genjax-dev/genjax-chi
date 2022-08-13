import jax
import jax.numpy as jnp
import genjax as gex
import pytest


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


def test_simple_normal_update(benchmark):
    key = jax.random.PRNGKey(314159)
    tr = jax.jit(gex.simulate(simple_normal))(key)
    new = {("y1",): 2.0}
    jitted = jax.jit(gex.update(simple_normal))
    w, updated = benchmark(jitted, tr, new, key)
    updated_chm = updated.get_choices()
    y1 = updated_chm[("y1",)]
    y2 = updated_chm[("y2",)]
    test_score = gex.Normal().score(y1) + gex.Normal().score(y2)
    assert updated.get_score() == pytest.approx(test_score, 0.01)
    original_score = tr.get_score()
    w, updated = jitted(tr, new, key)
    updated_chm = updated.get_choices()
    y1 = updated_chm[("y1",)]
    y2 = updated_chm[("y2",)]
    test_score = gex.Normal().score(y1) + gex.Normal().score(y2)
    assert updated.get_score() == original_score + w
    assert updated.get_score() == pytest.approx(test_score, 0.01)
