import jax
import jax.numpy as jnp
import genjax as gex
import pytest


def sample(key):
    key, y1 = gex.trace("y1", gex.Bernoulli)(key, 0.3)
    key, y2 = gex.trace("y2", gex.Beta)(key, 1, 1)
    return key, y1 + y2


def test_simple_normal():
    return
