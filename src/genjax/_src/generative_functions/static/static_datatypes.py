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


import jax.numpy as jnp  # noqa: I001
from genjax._src.core.datatypes.generative import (
    GenerativeFunction,
    HierarchicalChoiceMap,
    MapSelection,
    Trace,
)
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.serialization.pickle import (
    PickleDataFormat,
    PickleSerializationBackend,
    SupportsPickleSerialization,
)
from genjax._src.core.typing import Any, FloatArray, Tuple, dispatch

#########
# Trace #
#########


class StaticTrace(
    Trace,
    SupportsPickleSerialization,
):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    address_choices: Trie
    cache: Trie
    score: FloatArray

    def get_gen_fn(self):
        return self.gen_fn

    def get_choice(self):
        return HierarchicalChoiceMap(self.address_choices).strip()

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def get_subtrace(self, addr):
        return self.address_choices[addr]

    @dispatch
    def project(
        self,
        selection: MapSelection,
    ) -> FloatArray:
        weight = jnp.array(0.0)
        for k, subtrace in self.address_choices.get_submaps_shallow():
            if selection.has_addr(k):
                weight += subtrace.project(selection.get_subselection(k))
        return weight

    def has_cached_value(self, addr):
        return self.cache.has_submap(addr)

    def get_cached_value(self, addr):
        return self.cache.get_submap(addr)

    def get_aux(self):
        return (
            self.address_choices,
            self.cache,
        )

    #################
    # Serialization #
    #################

    @dispatch
    def dumps(
        self,
        backend: PickleSerializationBackend,
    ) -> PickleDataFormat:
        args, retval, score = self.args, self.retval, self.score
        choices_payload = []
        addr_payload = []
        for addr, subtrace in self.address_choices.get_submaps_shallow():
            inner_payload = subtrace.dumps(backend)
            choices_payload.append(inner_payload)
            addr_payload.append(addr)
        payload = [
            backend.dumps(args),
            backend.dumps(retval),
            backend.dumps(score),
            backend.dumps(addr_payload),
            backend.dumps(choices_payload),
        ]
        return PickleDataFormat(payload)
