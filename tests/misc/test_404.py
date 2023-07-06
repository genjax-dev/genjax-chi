# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import functools
from dataclasses import dataclass
import numpy as np
import pytest

import genjax
from genjax.typing import Callable, Any

_global = jnp.arange(3)

@genjax.gen
def localization_kernel(x):
    y = genjax.normal(jnp.sum(_global), 1.0) @ "x"
    return x + y

def wrap(fn):
    @genjax.gen
    def inner(carry, *static_args):
        idx, state = carry
        newstate = fn.inline(state, *static_args)
        return idx + 1, newstate
    
    return inner

class TestIssue404:
    def test_issue_404(self):
        key = jax.random.PRNGKey(314159)
        localization_chain = genjax.Unfold(wrap(localization_kernel), max_length = 3)
        _, trace = genjax.simulate(localization_chain)(key, (2, (0, 0.0)))
        assert True
