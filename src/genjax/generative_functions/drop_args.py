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
from genjax._src.core.datatypes.generative import Trace, GenerativeFunction, ChoiceMap
from genjax._src.core.typing import Tuple, Any, FloatArray, PRNGKey


@dataclass
class DroppedArgsTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    score: FloatArray
    inner_choice_map: ChoiceMap

    def flatten(self):
        return (
            self.gen_fn,
            self.args,
            self.retval,
            self.score,
            self.inner_choice_map,
        ), ()


@dataclass
class DroppedArgsGenerativeFunction(GenerativeFunction):
    gen_fn: GenerativeFunction
    keep: Tuple

    def flatten(self):
        return (self.gen_fn, self.keep), ()

    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> DroppedArgsTrace:
        tr = self.gen_fn.simulate(key, args)
        inner_args = tr.get_args()
        inner_retval = tr.get_retval()
        inner_chm = tr.get_choices()
        inner_score = tr.get_score()
        kept_args = inner_args[self.keep]
        return DroppedArgsTrace(self, kept_args, inner_retval, inner_chm, inner_score)

    def importance(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, DroppedArgsTrace]:
        w, tr = self.gen_fn.importance(key, choice_map, args)
        inner_args = tr.get_args()
        inner_retval = tr.get_retval()
        inner_chm = tr.get_choices()
        inner_score = tr.get_score()
        kept_args = inner_args[self.keep]
        return w, DroppedArgsTrace(
            self, kept_args, inner_retval, inner_chm, inner_score
        )

    # This is where the tricky Gen-breaking stuff happens.
    def update(
        self,
        key: PRNGKey,
        prev: DroppedArgsTrace,
        choice_map: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[Any, FloatArray, DroppedArgsTrace, ChoiceMap]:
        (retval_diff, w, tr, discard) = self.gen_fn.update(
            key, prev, choice_map, argdiffs
        )
        inner_args = tr.get_args()
        inner_retval = tr.get_retval()
        inner_chm = tr.get_choices()
        inner_score = tr.get_score()
        kept_args = inner_args[self.keep]
        return (
            retval_diff,
            w,
            DroppedArgsTrace(self, kept_args, inner_retval, inner_chm, inner_score),
            discard,
        )

    def assess(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, DroppedArgsTrace]:
        return self.gen_fn.assess(key, choice_map, args)
