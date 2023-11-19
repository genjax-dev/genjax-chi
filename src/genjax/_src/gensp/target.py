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
from genjax._src.core.datatypes.generative import ChoiceValue
from genjax._src.core.datatypes.generative import EmptyChoice
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters.forward import InitialStylePrimitive
from genjax._src.core.interpreters.forward import initial_style_bind
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


###################
# Score intrinsic #
###################

score_p = InitialStylePrimitive("score")


@typecheck
def accum_score(*args):
    (v,) = args
    initial_style_bind(score_p)(lambda v: v)(v)


##########
# Target #
##########


@dataclass
class Target(Pytree):
    p: GenerativeFunction
    args: Tuple
    constraints: ChoiceMap

    def flatten(self):
        return (self.p, self.args, self.constraints), ()

    def get_trace_type(self):
        inner_type = self.p.get_trace_type(*self.args)
        latent_selection = self.latent_selection()
        trace_type = inner_type.filter(latent_selection)
        return trace_type

    def latent_selection(self):
        return self.constraints.get_selection().complement()

    def get_latents(self, v):
        latent_selection = self.latent_selection()
        latents = v.strip().filter(latent_selection)
        return latents

    @dispatch
    def importance(self, key: PRNGKey, chm: ChoiceValue):
        inner = chm.get_leaf_value()
        assert isinstance(inner, ChoiceMap)
        merged = self.constraints.safe_merge(inner)
        (_, tr) = self.p.importance(key, merged, self.args)
        return (0.0, tr)

    @dispatch
    def importance(self, key: PRNGKey):
        (_, tr) = self.p.importance(key, self.constraints, self.args)
        return (0.0, tr)


##############
# Shorthands #
##############


@dispatch
def target(
    p: GenerativeFunction,
    args: Tuple,
):
    return Target.new(p, args, EmptyChoice())


@dispatch
def target(
    p: GenerativeFunction,
    args: Tuple,
    constraints: ChoiceMap,
):
    return Target.new(p, args, constraints)
