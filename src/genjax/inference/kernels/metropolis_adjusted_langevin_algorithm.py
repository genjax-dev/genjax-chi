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
import jax.numpy as jnp
import jax.tree_util as jtu
import tree_math as tm

from genjax.core.datatypes import Selection
from genjax.core.datatypes import Trace
from genjax.generative_functions.distributions.standard.normal import Normal
from genjax.inference.kernels.kernel import MCMCKernel


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jtu.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jtu.tree_unflatten(treedef, keys)


def tree_random_normal_fixed_std(rng_key, mu, std):
    keys_tree = random_split_like_tree(rng_key, mu)
    return jax.tree_map(
        lambda m, k: Normal.sample(k, m, std),
        mu,
        keys_tree,
    )


def tree_logpdf_normal_fixed_std(values, mu, std):
    vec = tm.Vector(
        jax.tree_map(
            lambda v, m: Normal.logpdf(v, m, std),
            values,
            mu,
        )
    )
    return vec.sum()


def _metropolis_adjusted_langevin_algorithm(selection: Selection, tau):
    def _inner(key, trace: Trace):
        args = trace.get_args()
        gen_fn = trace.get_gen_fn()
        std = jnp.sqrt(2 * tau)

        # Forward proposal.
        key, forward_gradient_trie, _ = gen_fn.choice_grad(
            key, trace, selection
        )
        forward_values, _ = selection.filter(trace)
        forward_values = forward_values.strip()
        forward_mu = jtu.tree_map(
            lambda v1, v2: v1 + tau * v2, forward_values, forward_gradient_trie
        )

        key, sub_key = jax.random.split(key)
        proposed_values = tree_random_normal_fixed_std(
            sub_key, forward_mu, std
        )
        forward_score = tree_logpdf_normal_fixed_std(
            proposed_values, forward_mu, std
        )

        # Get model weight.
        key, (weight, new_trace, _) = gen_fn.update(
            key, trace, proposed_values, args
        )

        # Backward proposal.
        key, backward_gradient_trie, _ = gen_fn.choice_grad(
            key, new_trace, selection
        )
        backward_mu = jtu.tree_map(
            lambda v1, v2: v1 + tau * v2,
            proposed_values,
            backward_gradient_trie,
        )
        backward_score = tree_logpdf_normal_fixed_std(
            forward_values, backward_mu, std
        )

        alpha = weight - forward_score + backward_score
        key, sub_key = jax.random.split(key)
        check = jnp.log(jax.random.uniform(sub_key)) < alpha
        return key, jax.lax.cond(
            check,
            lambda *args: (new_trace, True),
            lambda *args: (trace, False),
        )

    return _inner


metropolis_adjusted_langevin_algorithm = MCMCKernel(
    _metropolis_adjusted_langevin_algorithm
)
metropolis_adjusted_langevin_algorithm.set_reversal(
    metropolis_adjusted_langevin_algorithm
)

mala = metropolis_adjusted_langevin_algorithm
