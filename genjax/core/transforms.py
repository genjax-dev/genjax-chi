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
from genjax.core.handling import eval_jaxpr_handler

# Our special interpreter -- allows us to dispatch with primitives,
# and implements directed CPS-style code generation strategy.
def I_prime(handler_stack, f):
    return lambda *xs: eval_jaxpr_handler(
        handler_stack, f.jaxpr, f.literals, *xs
    )


def handle(handler_stack, expr):
    """
    Sugar: Abstract interpret a `Jaxpr` with a `handler_stack :: List Handler`
    """
    return I_prime(handler_stack, expr)


def is_concrete(x):
    return not isinstance(x, jax.core.Tracer)


def concrete_cond(pred, true_branch, false_branch, *args):
    if is_concrete(pred):
        if pred:
            return true_branch(*args)
        else:
            return false_branch(*args)
    else:
        return jax.lax.cond(pred, true_branch, false_branch, *args)
