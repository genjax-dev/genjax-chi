# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from genjax.core.pytree import Pytree
import genjax.core.pretty_printer as pp


@dataclass
class Trie(Pytree):
    nodes: dict

    def flatten(self):
        return (self.nodes,), ()

    @classmethod
    def unflatten(cls, xs, data):
        return Trie(*data)

    def is_empty(self):
        return len(self.nodes) == 0

    def has_node(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_node(first):
                subtrie = self.get_node(first)
                if not isinstance(subtrie, Trie):
                    return False
                return subtrie.has_node(rest)
            else:
                return False
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return addr in self.nodes

    def get_node(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_node(first):
                subtrie = self.get_node(first)
                if not isinstance(subtrie, Trie):
                    raise Exception(f"Found value at address {first}")
                return subtrie.get_node(rest)
            else:
                raise Exception(f"Trie has no subtree at {first}")
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return self.nodes[addr]

    def set_node(self, addr, value):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if not self.has_node(first):
                subtrie = Trie({})
                self.nodes[first] = subtrie
            subtrie = self.nodes[first]
            subtrie.set_node(rest, value)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            self.nodes[addr] = value

    def delete_node(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            if self.has_node(first):
                subtrie = self.get_node(first)
                if not isinstance(subtrie, Trie):
                    return False
                if subtrie.delete_node(rest):
                    self.delete_node(first)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            self.nodes.pop(addr, None)

        return len(self.nodes) == 0

    def has_choice(self, addr):
        return self.has_node(addr)

    def get_choice(self, addr):
        return self.get_node(addr)

    def get_choices_shallow(self):
        return self.nodes

    def merge(self, other):
        for (k, v) in other.get_choices_shallow():
            self.set_node(k, v)
        return self

    def __getitem__(self, addr):
        return self.get_node(addr)

    def __setitem__(self, addr, value):
        return self.set_node(addr, value)

    def __repr__(self):
        return pp.tree_pformat(self)

    def __str__(self):
        return pp.tree_pformat(self)
