import jax
import jax.numpy as jnp
import genjax as gex
import pytest


def f(key):
    key, y = gex.trace("y", gex.Normal)(key)
    return key, y


def test_f():
    key = jax.random.PRNGKey(314159)
    tr = jax.jit(gex.simulate(f))(key)
    _, v = tr.get_retval()
    assert tr.get_score() == pytest.approx(gex.Normal().score(v), 0.01)
