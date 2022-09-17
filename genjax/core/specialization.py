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

"""
This module holds some JAX-specific utilities to force
evaluation (where possible) or else create branch primitives
(e.g. :code:`jax.lax.switch`) when a value is not concrete.

These utilities are used throughout the codebase -- to gain some confidence
that tracing will actually collapse potential branches when values are known
statically.
"""

import jax
import jax.numpy as jnp
import operator
from functools import reduce


def is_concrete(x):
    return not isinstance(x, jax.core.Tracer)


def concrete_and(*args):
    if all(map(is_concrete, args)):
        return reduce(operator.and_, args, True)
    else:
        return jnp.logical_and(*args)


def concrete_cond(pred, true_branch, false_branch, *args):
    if is_concrete(pred):
        if pred:
            return true_branch(*args)
        else:
            return false_branch(*args)
    else:
        return jax.lax.cond(pred, true_branch, false_branch, *args)


def concrete_switch(index, branches, *args):
    if is_concrete(index):
        return branches[index](*args)
    else:
        return jax.lax.switch(index, branches, *args)
