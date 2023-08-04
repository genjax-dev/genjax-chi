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
from genjax._src.core.datatypes.generative import (
    Trace,
    GenerativeFunction,
    ChoiceMap,
)
from genjax._src.core.typing import Tuple, Any, FloatArray, PRNGKey


@dataclass
class DroppedArgumentsTrace(Trace):
    gen_fn: GenerativeFunction
    retval: Any
    score: FloatArray
    inner_choice_map: ChoiceMap
    aux: Tuple

    def flatten(self):
        return (
            self.gen_fn,
            self.retval,
            self.score,
            self.inner_choice_map,
            self.aux,
        ), ()


@dataclass
class DroppedArgumentsGenerativeFunction(GenerativeFunction):
    gen_fn: GenerativeFunction

    def flatten(self):
        return (self.gen_fn,), ()

    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> DroppedArgumentsTrace:
        tr = self.gen_fn.simulate(key, args)
        inner_retval = tr.get_retval()
        inner_chm = tr.get_choices()
        inner_score = tr.get_score()
        aux = tr.get_aux()
        return DroppedArgumentsTrace(self, inner_retval, inner_chm, inner_score, aux)

    def importance(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, DroppedArgumentsTrace]:
        w, tr = self.gen_fn.importance(key, choice_map, args)
        inner_retval = tr.get_retval()
        inner_chm = tr.get_choices()
        inner_score = tr.get_score()
        aux = tr.get_aux()
        return w, DroppedArgumentsTrace(self, inner_retval, inner_chm, inner_score, aux)

    def update(
        self,
        key: PRNGKey,
        prev: DroppedArgumentsTrace,
        choice_map: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[Any, FloatArray, DroppedArgumentsTrace, ChoiceMap]:
        raise NotImplementedError

    def assess(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, DroppedArgumentsTrace]:
        return self.gen_fn.assess(key, choice_map, args)
