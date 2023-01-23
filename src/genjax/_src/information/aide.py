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

"""This module contains an implementation of (Auxiliary inference divergence
estimator) from Cusumano-Towner et al, 2017."""

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey


@dataclasses.dataclass
class AuxiliaryInferenceDivergenceEstimator(Pytree):
    num_meta_p: Int
    num_meta_q: Int

    def flatten(self):
        return (), (self.num_meta_p, self.num_meta_q)

    def _estimate_log_ratio(
        self,
        key: PRNGKey,
        p: GenerativeFunction,
        p_args: Tuple,
        q: GenerativeFunction,
        q_args: Tuple,
    ):
        # Inner functions -- to be mapped over.
        # Keys are folded in, for working memory.
        def _inner_p(key, index, chm, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = p.importance(new_key, chm, args)
            return w

        def _inner_q(key, index, chm, args):
            new_key = jax.random.fold_in(key, index)
            _, (w, _) = q.importance(new_key, chm, args)
            return w

        key_indices_p = jnp.arange(0, self.num_meta_p + 1)
        key_indices_q = jnp.arange(0, self.num_meta_q + 1)

        key, tr = p.simulate(key, p_args)
        chm = tr.get_choices().strip()
        key, sub_key = jax.random.split(key)
        fwd_weights = jax.vmap(_inner_p, in_axes=(None, 0, None, None))(
            sub_key, key_indices_p, chm, p_args
        )
        key, sub_key = jax.random.split(key)
        bwd_weights = jax.vmap(_inner_q, in_axes=(None, 0, None, None))(
            sub_key, key_indices_q, chm, q_args
        )
        fwd_weight = logsumexp(fwd_weights) - jnp.log(self.num_meta_p)
        bwd_weight = logsumexp(bwd_weights) - jnp.log(self.num_meta_q)
        return key, fwd_weight - bwd_weight

    def estimate(
        self,
        key: PRNGKey,
        p: GenerativeFunction,
        p_args: Tuple,
        q: GenerativeFunction,
        q_args: Tuple,
    ):
        key, logpq = self._estimate_log_ratio(p, q, self.num_meta_p, self.num_meta_q)(
            key, p_args, q_args
        )
        key, logqp = self._estimate_log_ratio(q, p, self.num_meta_q, self.num_meta_p)(
            key, q_args, p_args
        )
        return key, logpq + logqp, (logpq, logqp)

    def __call__(
        self,
        key: PRNGKey,
        p: GenerativeFunction,
        p_args: Tuple,
        q: GenerativeFunction,
        q_args: Tuple,
    ):
        return self.estimate(key, p, p_args, q, q_args)


##############
# Shorthands #
##############

aide = AuxiliaryInferenceDivergenceEstimator.new
