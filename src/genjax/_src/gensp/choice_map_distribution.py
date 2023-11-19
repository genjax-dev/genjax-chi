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

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import ChoiceValue
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.gensp.sp_distribution import SPDistribution
from genjax._src.gensp.target import target


@dataclass
class ChoiceMapDistribution(SPDistribution):
    p: GenerativeFunction
    selection: Selection
    custom_q: Union[None, "ChoiceMapDistribution"]

    def flatten(self):
        return (), (self.p, self.selection, self.custom_q)

    @classmethod
    def new(cls, p: GenerativeFunction, selection=None, custom_q=None):
        if selection is None:
            selection = AllSelection()
        return ChoiceMapDistribution(p, selection, custom_q)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ):
        key, sub_key = jax.random.split(key)
        tgt = target(self.p, args)
        energy, tr = tgt.importance(sub_key)
        choices = tr.strip()
        selected_choices = choices.filter(self.selection)
        if self.custom_q is None:
            weight = energy + tr.project(self.selection)
            return (weight, ChoiceValue(selected_choices))
        else:
            key, sub_key = jax.random.split(key)
            unselected = choices.filter(self.selection.complement())
            tgt = target(self.p, args, selected_choices)
            w = self.custom_q.estimate_logpdf(key, ChoiceValue.new(unselected), tgt)
            weight = energy + tr.project(self.selection) - w
            return (weight, ChoiceValue(selected_choices))

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        choices: ChoiceValue,
        *args,
    ):
        inner_chm = choices.get_value()
        assert isinstance(inner_chm, ChoiceMap)
        if self.custom_q is None:
            tgt = target(self.p, args, inner_chm)
            (energy, tr) = tgt.importance(key)
            weight = energy + tr.project(self.selection)
            return weight
        else:
            key, sub_key = jax.random.split(key)
            tgt = target(self.p, args, inner_chm)
            (bwd_weight, other_choices) = self.custom_q.random_weighted(sub_key, tgt)
            (energy, tr) = tgt.importance(key, other_choices.get_value())
            fwd_weight = energy + tr.project(self.selection)
            weight = fwd_weight - bwd_weight
            return weight


##############
# Shorthands #
##############

choice_map_distribution = ChoiceMapDistribution.new
