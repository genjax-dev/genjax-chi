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

import abc
import functools
from dataclasses import dataclass

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import context as ctx
from genjax._src.core.interpreters import primitives
from genjax._src.core.interpreters.context import Context
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Float
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple


#########################
# Unnormalized measures #
#########################


@dataclass
class UnnormalizedMeasure(Pytree):
    @abc.abstractmethod
    def get_trace_type(self, *args):
        pass

    @abc.abstractmethod
    def latent_selection(self):
        pass

    @abc.abstractmethod
    def get_latents(self, choice_map):
        pass

    @abc.abstractmethod
    def importance(self, key: PRNGKey, chm: ChoiceMap, args: Tuple):
        pass


###################
# Score intrinsic #
###################

score_p = primitives.InitialStylePrimitive("score")


@score_p.def_impl
def _score_impl(*args, **_):
    return args


def score(*args, **kwargs):
    return primitives.initial_style_bind(score_p)(_score_impl)(*args, **kwargs)


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
class UnnormalizedMeasureFunction(UnnormalizedMeasure):
    gen_fn: GenerativeFunction

    def flatten(self):
        return (self.gen_fn,), ()

    def get_trace_type(self, *args):
        return self.gen_fn.get_trace_type(*args)

    def latent_selection(self):
        return AllSelection()

    def get_latents(self, v):
        return self.latent_selection().filter(v.strip())

    def importance(self, key: PRNGKey, chm: ChoiceMap, args: Tuple):
        importance_fn = self.gen_fn.importance
        rescaled = rescale_transform(importance_fn)
        (key, (w, tr)), energy = rescaled(key, chm, args)
        return key, (w + energy, tr)


##############
# Shorthands #
##############

lang = UnnormalizedMeasureFunction.new
