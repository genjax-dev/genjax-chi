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

import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax.core.datatypes import Selection
from genjax.core.datatypes import Trace
from genjax.core.typing import FloatTensor
from genjax.generative_functions.distributions.scipy.normal import Normal
from genjax.inference.mcmc.kernel import MCMCKernel


@dataclasses.dataclass
class MetropolisAdjustedLangevinAlgorithm(MCMCKernel):
    selection: Selection
    tau: FloatTensor

    def flatten(self):
        return (), (self.selection, self.tau)

    def _random_split_like_tree(self, rng_key, target=None, treedef=None):
        if treedef is None:
            treedef = jtu.tree_structure(target)
        keys = jax.random.split(rng_key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def _tree_random_normal_fixed_std(self, rng_key, mu, std):
        keys_tree = self._random_split_like_tree(rng_key, mu)
        return jax.tree_map(
            lambda m, k: Normal.sample(k, m, std),
            mu,
            keys_tree,
        )

    def _tree_logpdf_normal_fixed_std(self, values, mu, std):
        logpdf_tree = jax.tree_map(
            lambda v, m: Normal.logpdf(v, m, std),
            values,
            mu,
        )
        leaves = jtu.tree_leaves(logpdf_tree)
        return leaves.sum()

    def apply(self, key, trace: Trace):
        args = trace.get_args()
        gen_fn = trace.get_gen_fn()
        std = jnp.sqrt(2 * self.tau)

        # Forward proposal.
        key, forward_gradient_trie, _ = gen_fn.choice_grad(
            key, trace, self.selection, (0.0,)
        )
        forward_values, _ = self.selection.filter(trace)
        forward_values = forward_values.strip()
        forward_mu = jtu.tree_map(
            lambda v1, v2: v1 + self.tau * v2,
            forward_values,
            forward_gradient_trie,
        )

        key, sub_key = jax.random.split(key)
        proposed_values = self._tree_random_normal_fixed_std(
            sub_key, forward_mu, std
        )
        forward_score = self._tree_logpdf_normal_fixed_std(
            proposed_values, forward_mu, std
        )

        # Get model weight.
        key, (weight, new_trace, _) = gen_fn.update(
            key, trace, proposed_values, args
        )

        # Backward proposal.
        key, backward_gradient_trie, _ = gen_fn.choice_grad(
            key, new_trace, self.selection, (0.0,)
        )
        backward_mu = jtu.tree_map(
            lambda v1, v2: v1 + self.tau * v2,
            proposed_values,
            backward_gradient_trie,
        )
        backward_score = self._tree_logpdf_normal_fixed_std(
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

    def reversal(self):
        return self


mala = MetropolisAdjustedLangevinAlgorithm
