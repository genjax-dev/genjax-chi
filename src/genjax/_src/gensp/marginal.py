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

from dataclasses import dataclass

import jax

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.datatypes.generative import choice_map
from genjax._src.core.datatypes.generative import select
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_datatypes import (
    HierarchicalSelection,
)
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.gensp.choice_map_distribution import ChoiceMapDistribution
from genjax._src.gensp.target import Target


@dataclass
class Marginal(Distribution):
    p: GenerativeFunction
    q: ChoiceMapDistribution
    addr: Any

    def flatten(self):
        return (), (self.p, self.q, self.addr)

    def get_trace_type(self, *args):
        inner_type = self.p.get_trace_type(*args)
        selection = HierarchicalSelection.new([self.addr])
        trace_type = inner_type.filter(selection)
        return trace_type

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Any:
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, args)
        weight = tr.get_score()
        choices = tr.get_choices()
        val = choices[self.addr]
        selection = select(self.addr).complement()
        other_choices = choices.filter(selection)
        target = Target.new(self.p, args, choice_map({self.addr: val}))
        (q_weight, _) = self.q.importance(
            key, ValueChoiceMap.new(other_choices), (target,)
        )
        weight -= q_weight
        return (weight, val)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        val: Any,
        *args,
    ) -> FloatArray:
        chm = choice_map({self.addr: val})
        target = Target.new(self.p, args, chm)
        key, sub_key = jax.random.split(key)
        tr = self.q.simulate(sub_key, (target,))
        q_w = tr.get_score()
        choices = tr.get_choices()
        choices = choices.safe_merge(chm)
        (p_w, _) = self.p.importance(key, choices, args)
        return p_w - q_w


##############
# Shorthands #
##############

marginal = Marginal.new
