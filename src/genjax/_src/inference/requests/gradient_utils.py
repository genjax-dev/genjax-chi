# Copyright 2024 MIT Probabilistic Computing Project
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

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import grad
from tensorflow_probability.substrates import jax as tfp

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    Selection,
    Trace,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.typing import (
    Any,
    static_check_supports_grad,
)

tfd = tfp.distributions


# Pytree manipulation utilities -- these handle unzipping Pytrees into
# differentiable and non-diff pieces, and then also zipping them back up.
def grad_tree_unzip(tree: ChoiceMap) -> tuple[ChoiceMap, ChoiceMap]:
    grad_tree = jtu.tree_map(
        lambda v: v if static_check_supports_grad(v) else None, tree
    )
    nongrad_tree = jtu.tree_map(
        lambda v: v if not static_check_supports_grad(v) else None, tree
    )
    return grad_tree, nongrad_tree


def grad_tree_zip(
    grad_tree: ChoiceMap,
    nongrad_tree: ChoiceMap,
) -> ChoiceMap:
    return jtu.tree_map(
        lambda v1, v2: v1 if v1 is not None else v2, grad_tree, nongrad_tree
    )


# Compute the gradient of a selection of random choices
# in a trace -- uses `GenerativeFunction.assess`.
def selection_gradient(
    selection: Selection,
    trace: Trace[Any],
    argdiffs: Argdiffs,
) -> tuple[ChoiceMap, ChoiceMap]:
    chm = trace.get_choices()
    filtered = chm.filter(selection)
    complement = chm.filter(~selection)
    grad_tree, nongrad_tree = grad_tree_unzip(filtered)
    gen_fn = trace.get_gen_fn()

    def differentiable_assess(grad_tree):
        zipped = grad_tree_zip(grad_tree, nongrad_tree)
        full_choices = zipped.merge(complement)
        weight, _ = gen_fn.assess(
            full_choices,
            Diff.tree_primal(argdiffs),
        )
        return weight

    return grad_tree_zip(grad_tree, nongrad_tree), jtu.tree_map(
        lambda v1, v2: v1
        if v1 is not None
        else jnp.zeros_like(jnp.array(v2, copy=False)),
        grad(differentiable_assess)(grad_tree),
        nongrad_tree,
    )


__all__ = [
    "grad_tree_unzip",
    "grad_tree_zip",
    "selection_gradient",
]
