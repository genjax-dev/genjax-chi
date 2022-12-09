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

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu

from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Trace
from genjax.inference.kernels.kernel import MCMCKernel
from genjax.generative_functions.diff_rules import Diff


def _metropolis_hastings(proposal: GenerativeFunction):
    def _inner(key, trace: Trace, proposal_args: Tuple):
        model = trace.get_gen_fn()
        model_args = trace.get_args()
        proposal_args_fwd = (trace.get_choices(), *proposal_args)
        key, proposal_tr = proposal.simulate(key, proposal_args_fwd)
        fwd_weight = proposal_tr.get_score()
        diffs = jtu.tree_map(Diff.no_change, model_args)
        key, (_, weight, new, discard) = model.update(
            key, trace, proposal_tr.get_choices(), diffs
        )
        proposal_args_bwd = (new, *proposal_args)
        key, (bwd_weight, _) = proposal.importance(
            key, discard, proposal_args_bwd
        )
        alpha = weight - fwd_weight + bwd_weight
        key, sub_key = jax.random.split(key)
        check = jnp.log(random.uniform(sub_key)) < alpha
        return (
            key,
            jax.lax.cond(
                check,
                lambda *args: (new, True),
                lambda *args: (trace, False),
            ),
        )

    return _inner


metropolis_hastings = MCMCKernel(_metropolis_hastings)
metropolis_hastings.set_reversal(metropolis_hastings)
