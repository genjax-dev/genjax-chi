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

import functools
from dataclasses import dataclass

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import context as ctx
from genjax._src.core.interpreters import primitives
from genjax._src.core.interpreters.context import Context
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Float
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck


###################
# Score intrinsic #
###################

score_p = primitives.InitialStylePrimitive("score")


@typecheck
def accum_score(*args):
    (v,) = args
    primitives.initial_style_bind(score_p)(lambda v: v)(v)


#################################
# Unnormalized measure function #
#################################


@dataclass
class RescaleContext(Context):
    energy: Float

    def flatten(self):
        return (self.energy,), ()

    @classmethod
    def new(cls):
        return RescaleContext(0.0)

    def yield_state(self):
        return (self.energy,)

    def handle_score(self, _, tracer, **params):
        self.energy += tracer
        return [tracer]

    def can_process(self, primitive):
        return False

    def process_primitive(self, primitive):
        raise NotImplementedError

    def get_custom_rule(self, primitive):
        if primitive is score_p:
            return self.handle_score
        else:
            return None


def rescale_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(*args):
        context = RescaleContext.new()
        retvals, (energy,) = ctx.transform(source_fn, context)(*args, **kwargs)
        return retvals, energy

    return wrapper


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

    def importance(self, key, chm: ChoiceMap):
        merged = self.constraints.safe_merge(chm)
        importance_fn = self.p.importance
        rescaled = rescale_transform(importance_fn)
        (w, tr), energy = rescaled(key, merged, self.args)
        return (w + energy, tr)


##############
# Shorthands #
##############

target = Target.new
