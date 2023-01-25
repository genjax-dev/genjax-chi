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

from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.tracetypes import TraceType


#####
# VectorTraceType
#####


@dataclass
class VectorTraceType(TraceType):
    inner: TraceType
    length: int

    def flatten(self):
        return (), (self.inner, self.length)

    def is_leaf(self):
        return self.inner.is_leaf()

    def get_leaf_value(self):
        return self.inner.get_leaf_value()

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        v = self.inner.get_subtree(addr)
        return VectorTraceType(v, self.length)

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return (k, VectorTraceType(v, self.length))

        return map(lambda args: _inner(*args), self.inner.get_subtrees_shallow())

    def merge(self, other):
        raise Exception("Not implemented.")

    def __subseteq__(self, other):
        return False

    def get_rettype(self):
        return self.inner.get_rettype()

    def _tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        subk = Tree("[blue]length")
        subk.add(gpp.tree_pformat(self.length))
        subt = self.inner._build_rich_tree()
        tree.add(subk)
        tree.add(subt)
        return tree
