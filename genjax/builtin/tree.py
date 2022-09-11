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
from genjax.core.datatypes import ChoiceMap
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp


@dataclass
class Tree(ChoiceMap):
    nodes: dict

    def flatten(self):
        return (self.nodes,), ()

    @classmethod
    def unflatten(cls, xs, data):
        return Tree(*data)

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        text_list = []
        for (k, v) in self.get_choices_shallow():
            v = gpp._named_entry(k, v, **kwargs)
            text_list.append(v)
        return pp.concat(
            [
                pp.text("("),
                gpp._nest(indent, pp.join(gpp._comma_sep, text_list)),
                pp.brk(""),
                pp.text(")"),
            ]
        )

    def has_value(self):
        return False

    def get_value(self):
        raise Exception("Tree is not a value choice map.")

    def is_empty(self):
        return len(self.nodes) == 0

    def has_node(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_node(first):
                subtree = self.get_node(first)
                return subtree.has_choice(rest)
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
                subtree = self.get_node(first)
                return subtree.get_choice(rest)
            else:
                raise Exception(f"Tree has no subtree at {first}")
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return self.nodes[addr]

    def set_node(self, addr, value):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if not self.has_node(first):
                subtree = Tree({})
                self.nodes[first] = subtree
            subtree = self.nodes[first]
            subtree.set_node(rest, value)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            self.nodes[addr] = value

    def delete_node(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            if self.has_node(first):
                subtree = self.get_node(first)
                if not isinstance(subtree, Tree):
                    return False
                if subtree.delete_node(*rest):
                    self.delete_node(first)
        else:
            addr = addr[0]
            self.nodes.pop(addr, None)

        return len(self.nodes) == 0

    def has_choice(self, addr):
        return self.has_node(addr)

    def get_choice(self, addr):
        return self.get_node(addr)

    def get_choices_shallow(self):
        return self.nodes.items()

    def to_selection(self):
        tree = Tree({})
        for (k, v) in self.get_choices_shallow():
            tree[k] = v.to_selection()
        return tree

    def merge(self, other):
        tree = Tree({})
        for (k, v) in self.get_choices_shallow():
            if other.has_choice(k):
                sub = other[k]
                tree[k] = v.merge(sub)
            else:
                tree[k] = v
        for (k, v) in other.get_choices_shallow():
            if not self.has_choice(k):
                tree[k] = v
        return tree

    def __getitem__(self, addr):
        return self.get_node(addr)

    def __setitem__(self, addr, value):
        return self.set_node(addr, value)

    def __repr__(self):
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)

    def __hash__(self):
        hash_list = []
        for (k, v) in self.get_choices_shallow():
            hash_list.append((k, v))
        return hash(tuple(hash_list))
