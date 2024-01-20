# Copyright 2023 MIT Probabilistic Computing Project
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

The `DropArgumentsGenerativeFunction` exposes GFI methods which eliminate stored arguments in its returned trace. This is only valid if a caller `GenerativeFunction` which invokes a `DropArgumentsGenerativeFunction` provides arguments ("restores" the arguments) when it invokes `DropArgumentsGenerativeFunction` methods.

This is useful to avoid unnecessary allocations in e.g. `MapCombinator` which uses `jax.vmap` as part of its implementation, causing the arguments stored in its callee's trace to be expanded and stored (unnecessarily). `DropArgumentsGenerativeFunction` eliminates the stored arguments in the callee's trace -- and allows us to retain a single copy of the arguments in the `MapCombinator` caller's `MapTrace`.
"""


from genjax._src.core.datatypes.generative import (
    ChoiceMap,
    GenerativeFunction,
    JAXGenerativeFunction,
    Trace,
)
from genjax._src.core.typing import Any, FloatArray, PRNGKey, Tuple


class DropArgumentsTrace(Trace):
    gen_fn: GenerativeFunction
    retval: Any
    score: FloatArray
    inner_choice_map: ChoiceMap
    aux: Tuple

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

    def get_aux(self):
        return self.aux

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


class DropArgumentsGenerativeFunction(JAXGenerativeFunction):
    gen_fn: JAXGenerativeFunction

    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> DropArgumentsTrace:
        tr = self.gen_fn.simulate(key, args)
        inner_retval = tr.get_retval()
        inner_score = tr.get_score()
        inner_chm = tr.get_choices()
        aux = tr.get_aux()
        return DropArgumentsTrace(
            self,
            inner_retval,
            inner_score,
            inner_chm,
            aux,
        )

    def importance(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[DropArgumentsTrace, FloatArray]:
        w, tr = self.gen_fn.importance(key, choice_map, args)
        inner_retval = tr.get_retval()
        inner_score = tr.get_score()
        inner_chm = tr.get_choices()
        aux = tr.get_aux()
        return (
            DropArgumentsTrace(
                self,
                inner_retval,
                inner_score,
                inner_chm,
                aux,
            ),
            w,
        )

    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        choice_map: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[DropArgumentsTrace, FloatArray, Any, ChoiceMap]:
        (tr, w, retval_diff, discard) = self.gen_fn.update(
            key, prev, choice_map, argdiffs
        )
        inner_retval = tr.get_retval()
        inner_score = tr.get_score()
        inner_chm = tr.get_choices()
        aux = tr.get_aux()
        return (
            DropArgumentsTrace(
                self,
                inner_retval,
                inner_score,
                inner_chm,
                aux,
            ),
            w,
            retval_diff,
            discard,
        )

    def assess(
        self,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        return self.gen_fn.assess(choice_map, args)

    def restore_with_aux(self, interface_data, aux):
        return self.gen_fn.restore_with_aux(interface_data, aux)


#############
# Decorator #
#############

drop_arguments = DropArgumentsGenerativeFunction
