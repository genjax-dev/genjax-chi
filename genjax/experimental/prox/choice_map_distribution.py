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
from genjax.core.datatypes import GenerativeFunction, Selection
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target
from genjax.distributions.distribution import ValueChoiceMap


@dataclass
class ChoiceMapDistribution(ProxDistribution):
    p: GenerativeFunction
    selection: Selection
    custom_q: Any

    def random_weighted(self, key, *args):
        key, tr = self.p.simulate(key, args)
        choices = tr.get_choices()
        selected_choices = self.selection.filter(choices)
        if self.custom_q == None:
            weight = tr.project(self.selection)
        else:
            unselected = self.selection.complement.filter(choices)
            target = Target(self.p, args, selected_choices)
            key, (w, _) = self.custom_q.importance(
                key, ValueChoiceMap(unselected), (target,)
            )
            weight = tr.get_score() - w
        return key, (selected_choices, weight)

    def estimate_logpdf(self, key, choices, *args):
        if self.custom_q == None:
            key, (weight, _) = self.p.importance(key, choices, args)
        else:
            target = Target(self.p, args, choices)
            key, tr = self.custom_q.simulate(key, (target,))
            key, (w, _) = target.importance(key, tr.get_retval(), ())
            weight = w - tr.get_score()
        return key, weight
