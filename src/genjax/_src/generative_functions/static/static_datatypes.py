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

import rich.tree as rich_tree

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.serialization.pickle import PickleDataFormat
from genjax._src.core.serialization.pickle import PickleSerializationBackend
from genjax._src.core.serialization.pickle import SupportsPickleSerialization
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


#######################
# Dynamic choice map  #
#######################


@dataclass
class StaticLanguageChoiceMap(ChoiceMap):
    addrs: List[Any]
    subtraces: List[Trace]

    def flatten(self):
        return (self.addrs, self.submaps), ()

    @classmethod
    @dispatch
    def new(cls, addrs, submaps):
        return StaticLanguageChoiceMap(addrs, submaps)

    @classmethod
    @dispatch
    def new(cls):
        return StaticLanguageChoiceMap([], [])

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](StaticLanguageChoiceMap)")
        for k, v in zip(self.addrs, self.submaps):
            if isinstance(k, Pytree):
                subk = k.__rich_tree__()
            else:
                subk = rich_tree.Tree(f"[bold]{k}")

            subv = v.__rich_tree__()
            subk.add(subv)
            tree.add(subk)
        return tree


#########
# Trace #
#########


@dataclass
class StaticTrace(
    Trace,
    SupportsPickleSerialization,
):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    address_choices: StaticLanguageChoiceMap
    cache: Trie
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.args,
            self.retval,
            self.address_choices,
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
        address_choices: ChoiceMap,
        cache: Trie,
        score: FloatArray,
    ):
        return StaticTrace(
            gen_fn,
            args,
            retval,
            address_choices,
            cache,
            score,
        )

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return self.address_choices

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
        for k, subtrace in self.static_address_choices.get_submaps_shallow():
            if selection.has_submap(k):
                weight += subtrace.project(selection.get_submap(k))
        return weight

    def has_cached_value(self, addr):
        return self.cache.has_submap(addr)

    def get_cached_value(self, addr):
        return self.cache.get_submap(addr)

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
        for addr, subtrace in self.static_address_choices.get_submaps_shallow():
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
