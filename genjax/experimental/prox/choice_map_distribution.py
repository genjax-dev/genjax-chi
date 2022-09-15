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
from genjax.core.datatypes import GenerativeFunction, Selection, AllSelection
from genjax.experimental.prox.prox_distribution import ProxDistribution
from genjax.experimental.prox.target import Target
from genjax.distributions.distribution import ValueChoiceMap
from typing import Union


@dataclass
class ChoiceMapDistribution(ProxDistribution):
    p: GenerativeFunction
    selection: Selection
    custom_q: Union[None, ProxDistribution]

    def __init__(self, p, selection, custom_q=None):
        self.p = p
        self.selection = selection
        self.custom_q = custom_q

    @classmethod
    def new(cls, p: GenerativeFunction, selection=None):
        if selection is None:
            selection = AllSelection()
        return ChoiceMapDistribution(p, selection, None)

    def get_trace_type(self, key, args, **kwargs):
        inner_type = self.p.get_trace_type(key, args)
        trace_type, _ = self.selection.filter(inner_type)
        correct_if_check = trace_type
        if self.custom_q is None:
            return correct_if_check
        else:
            target = Target(self.p, args, self.selection)
            proposal_trace_type = self.custom_q.get_trace_type(key, (target,))
            target_trace_type = target.get_trace_type(key)
            check, mismatch = target_trace_type.subseteq(proposal_trace_type)
            if not check:
                raise Exception(
                    f"Trace type mismatch.\n{target} with proposal {self.custom_q}"
                    f"\n\nMeasure ⊆ failure at the following addresses:\n{mismatch}"
                )
            return correct_if_check

    def random_weighted(self, key, *args):
        key, tr = self.p.simulate(key, args)
        choices = tr.get_choices()
        selected_choices, _ = self.selection.filter(choices)
        if self.custom_q is None:
            _, weight = self.selection.filter(tr)
        else:
            unselected, _ = self.selection.complement().filter(choices)
            target = Target(self.p, args, selected_choices)

            # Check trace type.
            proposal_trace_type = self.custom_q.get_trace_type(key, (target,))
            target_trace_type = target.get_trace_type(key)
            check, mismatch = target_trace_type.subseteq(proposal_trace_type)
            if not check:
                raise Exception(
                    f"Trace type mismatch.\n{target} with proposal {self.custom_q}"
                    f"\n\nMeasure ⊆ failure at the following addresses:\n{mismatch}"
                )

            # Proceed.
            key, (w, _) = self.custom_q.importance(
                key, ValueChoiceMap(unselected), (target,)
            )
            weight = tr.get_score() - w
        return key, (selected_choices, weight)

    def estimate_logpdf(self, key, choices, *args):
        if self.custom_q is None:
            key, (weight, _) = self.p.importance(key, choices, args)
        else:
            target = Target(self.p, args, choices)

            # Check trace type.
            proposal_trace_type = self.custom_q.get_trace_type(key, (target,))
            target_trace_type = target.get_trace_type(key)
            check, mismatch = target_trace_type.subseteq(proposal_trace_type)
            if not check:
                raise Exception(
                    f"Trace type mismatch.\n{target} with proposal {self.custom_q}"
                    f"\n\nMeasure ⊆ failure at the following addresses:\n{mismatch}"
                )

            # Proceed.
            key, tr = self.custom_q.simulate(key, (target,))
            key, (w, _) = target.importance(key, tr.get_retval(), ())
            weight = w - tr.get_score()
        return key, weight
