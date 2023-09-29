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
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.gensp.target import Target


@dataclass
class ChoiceMapDistribution(Distribution):
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

    def get_trace_type(self, *args):
        raise NotImplementedError

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ):
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, args)
        choices = tr.get_choices()
        selected_choices = choices.filter(self.selection)
        if self.custom_q is None:
            unselected = choices.filter(self.selection.complement())
            target = Target.new(self.p, args, selected_choices)
            (weight, _) = target.importance(key, unselected)
            return (weight, ValueChoiceMap(selected_choices))
        else:
            unselected = choices.filter(self.selection.complement())
            target = Target.new(self.p, args, selected_choices)

            w = self.custom_q.estimate_logpdf(
                key, ValueChoiceMap.new(unselected), target
            )
            weight = tr.get_score() - w
            return (weight, ValueChoiceMap(selected_choices))

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        choices: ValueChoiceMap,
        *args,
    ):
        inner_chm = choices.get_leaf_value()
        assert isinstance(inner_chm, ChoiceMap)
        if self.custom_q is None:
            target = Target.new(self.p, args, inner_chm)
            (weight, _) = target.importance(key, EmptyChoiceMap())
            return weight
        else:
            key, sub_key = jax.random.split(key)
            target = Target.new(self.p, args, inner_chm)
            (bwd_weight, other_choices) = self.custom_q.random_weighted(sub_key, target)
            (fwd_weight, _) = target.importance(key, other_choices.get_leaf_value())
            weight = fwd_weight - bwd_weight
            return weight


##############
# Shorthands #
##############

choice_map_distribution = ChoiceMapDistribution.new
