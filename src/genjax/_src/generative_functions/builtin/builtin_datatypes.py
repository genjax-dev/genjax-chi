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

import jax.numpy as jnp

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalChoiceMap
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.serialization.pickle import PickleDataFormat
from genjax._src.core.serialization.pickle import PickleSerializationBackend
from genjax._src.core.serialization.pickle import SupportsPickleSerialization
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


########################################################
# Dynamic choice map (for dynamic address uncertainty) #
########################################################


@dataclass
class DynamicChoiceMap(ChoiceMap):
    dynamic_indices: List[IntArray]
    subtrees: List[ChoiceMap]

    def flatten(self):
        return (self.dynamic_indices, self.subtrees), ()

    @dispatch
    def get_subtree(self, addr: IntArray):
        pass

    @dispatch
    def get_subtree(self, addr: Tuple):
        (idx, rest) = addr
        disjoint_union_chm = self.get_subtree(idx)
        return disjoint_union_chm.get_subtree(rest)

    @dispatch
    def has_subtree(self, addr: IntArray):
        equality_checks = jnp.array(map(lambda v: addr == v, self.dynamic_indices))
        return jnp.any(equality_checks)

    @dispatch
    def has_subtree(self, addr: Tuple):
        (idx, rest) = addr
        disjoint_union_chm = self.get_subtree(idx)
        return jnp.logical_and(
            self.has_subtree(idx),
            disjoint_union_chm.has_subtree(rest),
        )


#########
# Trace #
#########


@dataclass
class BuiltinTrace(
    Trace,
    SupportsPickleSerialization,
):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Trie
    cache: Trie
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.args,
            self.retval,
            self.choices,
            self.cache,
            self.score,
        ), ()

    @typecheck
    @classmethod
    def new(
        cls,
        gen_fn: GenerativeFunction,
        args: Tuple,
        retval: Any,
        choices: Trie,
        cache: Trie,
        score: FloatArray,
    ):
        return BuiltinTrace(gen_fn, args, retval, choices, cache, score)

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return HierarchicalChoiceMap(self.choices)

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    @dispatch
    def project(
        self,
        selection: HierarchicalSelection,
    ) -> FloatArray:
        weight = 0.0
        for (k, subtrace) in self.choices.get_subtrees_shallow():
            if selection.has_subtree(k):
                weight += subtrace.project(selection.get_subtree(k))
        return weight

    def has_cached_value(self, addr):
        return self.cache.has_subtree(addr)

    def get_cached_value(self, addr):
        return self.cache.get_subtree(addr)

    def get_aux(self):
        return (self.cache,)

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
        for (addr, subtrace) in self.choices.get_subtrees_shallow():
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
