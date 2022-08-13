import jax
import jax.numpy as jnp
import genjax as gex
import pytest


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


def test_simple_normal_diff(benchmark):
    key = jax.random.PRNGKey(314159)
    tr = jax.jit(gex.simulate(simple_normal))(key)
    original = tr.get_choices().get(("y1, "))
    new = {("y1",): 2.0}
    jitted = jax.jit(gex.diff(simple_normal))
    w, ret = benchmark(jitted, tr, new, key)
