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

import itertools
from dataclasses import dataclass
from typing import Sequence

from genjax._src.core.tracetypes import TraceType


#####
# SumTraceType
#####


@dataclass
class SumTraceType(TraceType):
    summands: Sequence[TraceType]

    def flatten(self):
        return (), (self.summands,)

    def is_leaf(self):
        return all(map(lambda v: v.is_leaf(), self.summands))

    def get_leaf_value(self):
        pass

    def has_subtree(self, addr):
        return any(map(lambda v: v.has_subtree(addr), self.summands))

    def get_subtree(self, addr):
        pass

    def get_subtrees_shallow(self):
        sub_iterators = map(
            lambda v: v.get_subtrees_shallow(),
            self.summands,
        )
        return itertools.chain(*sub_iterators)

    def merge(self, other):
        raise Exception("Not implemented.")

    def __subseteq__(self, other):
        return False

    def get_rettype(self):
        return self.summands[0].get_rettype()
