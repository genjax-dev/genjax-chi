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
"""This module implements a custom type of `GenerativeFunction` designed to
support data allocation optimizations for special `GenerativeFunction` types
which support a notion of calling another `GenerativeFunction`. Examples of
this type include `MapCombinator`.

The `DroppedArgumentsGenerativeFunction` exposes GFI methods which eliminate stored arguments in its returned trace. This is only valid if a caller `GenerativeFunction` which invokes a `DroppedArgumentsGenerativeFunction` provides arguments ("restores" the arguments) when it invokes `DroppedArgumentsGenerativeFunction` methods.

This is useful to avoid unnecessary allocations in e.g. `MapCombinator` which uses `jax.vmap` as part of its implementation, causing the arguments stored in its callee's trace to be expanded and stored (unnecessarily). `DroppedArgumentsGenerativeFunction` eliminates the stored arguments in the callee's trace -- and allows us to retain a single copy of the arguments in the `MapCombinator` caller's `MapTrace`.
"""

from dataclasses import dataclass

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple


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

    def get_gen_fn(self):
        return self.gen_fn

    def get_args(self):
        return ()

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_choices(self):
        return self.inner_choice_map

    def restore(self, original_arguments: Tuple):
        interface_data = (
            original_arguments,
            self.get_retval(),
            self.get_score(),
            self.get_choices(),
        )
        aux = self.get_aux()
        restored = self.gen_fn.restore_with_aux(interface_data, aux)
        return restored


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
