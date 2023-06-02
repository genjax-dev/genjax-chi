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

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.generative_functions.distributions.gensp.unnorm import (
    UnnormalizedMeasure,
)
from genjax._src.generative_functions.distributions.gensp.unnorm import (
    UnnormalizedMeasureFunction,
)


@dataclass
class Target(UnnormalizedMeasure):
    p: Union[GenerativeFunction, UnnormalizedMeasureFunction]
    args: Tuple
    constraints: ChoiceMap

    def flatten(self):
        return (self.p, self.args, self.constraints), ()

    def get_trace_type(self):
        inner_type = self.p.get_trace_type(*self.args)
        latent_selection = self.latent_selection()
        trace_type = latent_selection.filter(inner_type)
        return trace_type

    def latent_selection(self):
        return self.constraints.get_selection().complement()

    def get_latents(self, v):
        latent_selection = self.latent_selection()
        latents = latent_selection.filter(v.strip())
        return latents

    def importance(self, key, chm: ChoiceMap, _: Tuple):
        merged = self.constraints.merge(chm)
        return self.p.importance(key, merged, self.args)


##############
# Shorthands #
##############

target = Target.new
