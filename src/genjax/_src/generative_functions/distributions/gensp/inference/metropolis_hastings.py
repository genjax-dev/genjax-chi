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

from genjax._src.core.datatypes import Trace
from genjax._src.generative_functions.builtin.propagating import Diff
from genjax._src.generative_functions.builtin.propagating import NoChange
from genjax._src.gensp.gensp_distribution import GenSPDistribution


def metropolis_hastings(proposal: GenSPDistribution):
    def _inner(key, trace: Trace, proposal_args: Tuple):
        model = trace.get_gen_fn()
        model_args = trace.get_args()
        diffs = tuple(map(lambda v: Diff.new(v, change=NoChange), model_args))
        key, tr = proposal.simulate(key, (trace, *proposal_args))
        fwd_weight = tr.get_score()
        choices = tr.get_choices()
        key, (weight, new, discard) = model.update(key, trace, choices, diffs)
        proposal_args_bwd = (new, *proposal_args)
        key, (bwd_weight, _) = proposal.importance(key, discard, proposal_args_bwd)
        alpha = weight - fwd_weight + bwd_weight
        key, sub_key = jax.random.split(key)
        check = jnp.log(jax.random.uniform(sub_key)) < alpha
        return (
            key,
            jax.lax.cond(
                check,
                lambda *args: (new, True),
                lambda *args: (trace, False),
            ),
        )

    return _inner
