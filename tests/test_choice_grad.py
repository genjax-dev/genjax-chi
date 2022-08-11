import jax
import jax.numpy as jnp
import genjax as gex
import pytest


def simple_normal(key):
    key, y1 = gex.trace("y1", gex.Normal)(key)
    key, y2 = gex.trace("y2", gex.Normal)(key)
    return key, y1 + y2


def test_simple_normal():
    key = jax.random.PRNGKey(314159)
    v1 = 0.5
    v2 = -0.5
    chm = {("y1",): v1, ("y2",): v2}
    tr = jax.jit(gex.simulate(simple_normal))(key)
    choice_grads, _ = jax.jit(gex.choice_grad(simple_normal))(tr, chm, key)
    test_grad_y1 = jax.grad(lambda v1: gex.Normal().score(v1))(v1)
    test_grad_y2 = jax.grad(lambda v2: gex.Normal().score(v2))(v2)
    assert choice_grads[("y1",)] == test_grad_y1
    assert choice_grads[("y2",)] == test_grad_y2
