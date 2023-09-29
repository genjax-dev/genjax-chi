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
from adevjax import ADEVPrimitive
from adevjax import flip_enum
from adevjax import sample_with_key

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.typing import Callable
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.generative_functions.distributions.distribution import ExactDensity
from genjax._src.generative_functions.distributions.scipy.bernoulli import bernoulli
from genjax._src.gensp.target import Target


@dataclass
class ADEVDistribution(ExactDensity):
    differentiable_logpdf: Callable
    adev_primitive: ADEVPrimitive

    def flatten(self):
        return (self.adev_primitive,), (self.differentiable_logpdf,)

    @classmethod
    def new(cls, adev_prim, diff_logpdf):
        return ADEVDistribution(diff_logpdf, adev_prim)

    def sample(self, key, *args):
        return sample_with_key(self.adev_primitive, key, args)

    def logpdf(self, v, *args):
        return self.differentiable_logpdf(v, *args)


flip_enum = ADEVDistribution.new(
    flip_enum,
    lambda v, p: bernoulli.logpdf(v, p),
)


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

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ):
        tr = self.p.simulate(key, args)
        choices = tr.get_choices()
        selected_choices = choices.filter(self.selection)
        if self.custom_q is None:
            weight = tr.project(self.selection)
            return (weight, ValueChoiceMap(selected_choices))
        else:
            unselected = choices.filter(self.selection.complement())
            target = Target.new(self.p, args, selected_choices)
            w = self.custom_q.estimate_logpdf(
                key, ValueChoiceMap.new(unselected), (target,)
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
        inner_choices = choices.get_leaf_value()
        assert isinstance(inner_choices, ChoiceMap)
        if self.custom_q is None:
            (_, w) = self.p.assess(key, inner_choices, args)
            return w
        else:
            target = Target(self.p, args, inner_choices)
            key, sub_key = jax.random.split(key)
            tr = self.custom_q.simulate(sub_key, (target,))
            (w, _) = target.importance(key, tr.get_retval(), ())
            weight = w - tr.get_score()
            return weight


##############
# Shorthands #
##############

chm_dist = ChoiceMapDistribution.new
target = Target.new
