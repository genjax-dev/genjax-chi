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
from typing import Callable
from typing import Tuple
from typing import Union

from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Trace
from genjax.core.pytree import Pytree


@dataclass
class Target(Pytree):
    p: GenerativeFunction
    choice_map_coercion: Union[None, Callable]
    args: Tuple
    constraints: ChoiceMap

    def flatten(self):
        return (self.args, self.constraints), (
            self.p,
            self.choice_map_coercion,
        )

    def get_trace_type(self, key, *args):
        inner_type = self.p.get_trace_type(key, self.args)
        latent_selection = self.latent_selection()
        trace_type, _ = latent_selection.filter(inner_type)
        return trace_type

    def latent_selection(self):
        return self.constraints.get_selection().complement()

    def get_latents(self, v):
        if isinstance(v, ChoiceMap):
            latents, _ = self.latent_selection().filter(v)
            return latents
        elif isinstance(v, Trace):
            latents, _ = self.latent_selection().filter(
                v.get_choices().strip()
            )
            return latents

    def importance(self, key, chm: ChoiceMap, _: Tuple):
        if self.choice_map_coercion is not None:
            chm = self.choice_map_coercion(self, chm)
        merged = self.constraints.merge(chm)
        return self.p.importance(key, merged, self.args)
