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
from typing import Any

from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import ValueChoiceMap
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target
from genjax.generative_functions.builtin.builtin_datatypes import (
    BuiltinChoiceMap,
)
from genjax.generative_functions.builtin.builtin_datatypes import (
    BuiltinSelection,
)
from genjax.generative_functions.distributions.distribution import Distribution


@dataclass
class Marginal(ProxDistribution):
    p: GenerativeFunction
    q: Distribution
    addr: Any

    def flatten(self):
        return (), (self.p, self.q, self.addr)

    def get_trace_type(self, key, *args):
        inner_type = self.p.get_trace_type(key, *args)
        selection = BuiltinSelection([self.addr])
        trace_type, _ = selection.filter(inner_type)
        return trace_type

    def random_weighted(self, key, *args):
        key, tr = self.p.simulate(key, args)
        weight = tr.get_score()
        choices = tr.get_choices()
        val = choices[self.addr]
        selection = BuiltinSelection([self.addr]).complement()
        other_choices, _ = selection.filter(choices)
        target = Target(self.p, args, BuiltinChoiceMap({self.addr: val}))
        key, (q_weight, _) = self.q.importance(
            key, ValueChoiceMap(other_choices), (target,)
        )
        weight -= q_weight
        return key, (weight, val)

    def estimate_logpdf(self, key, val, *args):
        chm = BuiltinChoiceMap({self.addr: val})
        target = Target(self.p, args, chm)
        key, tr = self.q.simulate(key, (target,))
        q_w = tr.get_score()
        choices = tr.get_choices()
        choices = choices.merge(chm)
        key, (p_w, _) = self.p.importance(key, choices, args)
        return key, (p_w - q_w, val)
