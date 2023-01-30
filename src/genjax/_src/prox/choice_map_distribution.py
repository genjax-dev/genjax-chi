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
from typing import Union

from genjax._src.core.datatypes import AllSelection
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import ValueChoiceMap
from genjax._src.prox.prox_distribution import ProxDistribution
from genjax._src.prox.target import Target
from genjax._src.prox.utils import stack_check_supports


@dataclass
class ChoiceMapDistribution(ProxDistribution):
    p: GenerativeFunction
    selection: Selection
    custom_q: Union[None, ProxDistribution]

    def flatten(self):
        return (), (self.p, self.selection, self.custom_q)

    @classmethod
    def new(cls, p: GenerativeFunction, selection=None, custom_q=None):
        if selection is None:
            selection = AllSelection()
        return ChoiceMapDistribution(p, selection, custom_q)

    def get_trace_type(self, *args):
        inner_type = self.p.get_trace_type(*args)
        trace_type = self.selection.filter(inner_type)
        correct_if_check = trace_type
        if self.custom_q is None:
            return correct_if_check
        else:
            target = Target(self.p, args, self.selection)
            stack_check_supports(target, self.custom_q)
            return correct_if_check

    def random_weighted(self, key, *args):
        key, tr = self.p.simulate(key, args)
        choices = tr.get_choices()
        selected_choices = self.selection.filter(choices)
        if self.custom_q is None:
            weight = tr.project(self.selection)
        else:
            unselected = self.selection.complement().filter(choices)
            target = Target(self.p, args, choices)

            # Perform a compile-time trace type check.
            stack_check_supports(target, self.custom_q)

            key, (w, _) = self.custom_q.assess(
                key, ValueChoiceMap.new(unselected), (target,)
            )
            weight = tr.get_score() - w
        return key, (weight, selected_choices)

    def estimate_logpdf(self, key, choices, *args):
        if self.custom_q is None:
            key, (_, weight) = self.p.assess(key, choices, args)
        else:
            target = Target(self.p, args, choices)

            # Perform a compile-time trace type check.
            stack_check_supports(target, self.custom_q)

            key, tr = self.custom_q.simulate(key, (target,))
            key, (w, _) = target.importance(key, tr.get_retval(), ())
            weight = w - tr.get_score()
        return key, weight


##############
# Shorthands #
##############

chm_dist = ChoiceMapDistribution.new
