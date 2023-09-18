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
from genjax._src.core.datatypes.generative import DisjointUnionChoiceMap
from genjax._src.core.datatypes.generative import DynamicHierarchicalChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalChoiceMap
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import IndexedChoiceMap
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.static_checks import (
    static_check_tree_structure_equivalence,
)
from genjax._src.core.pytree.utilities import tree_stack
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
    static_address_choices: Trie
    dynamic_addresses: List[IntArray]
    dynamic_address_choices: List[ChoiceMap]
    cache: Trie
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.args,
            self.retval,
            self.static_address_choices,
            self.dynamic_addresses,
            self.dynamic_address_choices,
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
        static_address_choices: Trie,
        dynamic_addresses: List[IntArray],
        dynamic_address_choices: List[Pytree],
        cache: Trie,
        score: FloatArray,
    ):
        return BuiltinTrace(
            gen_fn,
            args,
            retval,
            static_address_choices,
            dynamic_addresses,
            dynamic_address_choices,
            cache,
            score,
        )

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        # Handle coercion of the static address choices (a `Trie`)
        # to a choice map.
        if self.static_address_choices.is_empty():
            static_chm = EmptyChoiceMap()
        else:
            static_chm = HierarchicalChoiceMap.new(self.static_address_choices)

        # Now deal with the dynamic address choices.
        if not self.dynamic_addresses and not self.dynamic_address_choices:
            return static_chm
        else:
            # Specialized path: all structure is the same, we can coerce into
            # an `IndexedChoiceMap`.
            if static_check_tree_structure_equivalence(self.dynamic_address_choices):
                index_arr = jnp.stack(self.dynamic_addresses)
                stacked_inner = tree_stack(self.dynamic_address_choices)
                if isinstance(stacked_inner, Trie):
                    inner = HierarchicalChoiceMap.new(stacked_inner)
                else:
                    inner = stacked_inner
                dynamic = IndexedChoiceMap.new(index_arr, inner)

            # Fallback path: heterogeneous structure, we defer specialization
            # to other methods.
            else:
                dynamic = DynamicHierarchicalChoiceMap.new(
                    self.dynamic_addresses,
                    self.dynamic_address_choices,
                )

            if isinstance(static_chm, EmptyChoiceMap):
                return dynamic
            else:
                return DisjointUnionChoiceMap.new([static_chm, dynamic])

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
        for k, subtrace in self.static_address_choices.get_subtrees_shallow():
            if selection.has_subtree(k):
                weight += subtrace.project(selection.get_subtree(k))
        return weight

    def has_cached_value(self, addr):
        return self.cache.has_subtree(addr)

    def get_cached_value(self, addr):
        return self.cache.get_subtree(addr)

    def get_aux(self):
        return (
            self.static_address_choices,
            self.dynamic_addresses,
            self.dynamic_address_choices,
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
        for addr, subtrace in self.static_address_choices.get_subtrees_shallow():
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
