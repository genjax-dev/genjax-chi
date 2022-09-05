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
import jax.random as random
from genjax.core.datatypes import GenerativeFunction, Trace
from typing import Tuple


def metropolis_hastings(proposal: GenerativeFunction):
    def _inner(key, trace: Trace, proposal_args: Tuple):
        model = trace.get_gen_fn()
        model_args = trace.get_args()
        proposal_args_fwd = (trace.get_choices(), *proposal_args)
        key, proposal_tr = proposal.simulate(key, proposal_args_fwd)
        fwd_weight = proposal_tr.get_score()
        key, (weight, new, discard) = model.update(
            key, trace, proposal_tr.get_choices(), model_args
        )
        proposal_args_bwd = (new, *proposal_args)
        key, (bwd_weight, _) = proposal.importance(
            key, discard, proposal_args_bwd
        )
        alpha = weight - fwd_weight + bwd_weight
        check = jnp.log(random.uniform(key)) < alpha
        return key, jax.lax.cond(
            check,
            lambda *args: (new, True),
            lambda *args: (trace, False),
        )

    return _inner
