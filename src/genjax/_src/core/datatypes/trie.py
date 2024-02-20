# Copyright 2024 MIT Probabilistic Computing Project
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

import copy

import rich

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.hashable_dict import HashableDict, hashable_dict
from genjax._src.core.pretty_printing import CustomPretty
from genjax._src.core.pytree import Pytree

########
# Trie #
########


class Trie(Pytree, CustomPretty):
    inner: HashableDict = Pytree.field(default_factory=hashable_dict)

    def is_static_empty(self):
        return not bool(self.inner)

    def get_selection(self):
        raise Exception("Trie doesn't provide conversion to Selection.")

    # Returns a new `Trie` with shallow copied inner dictionary.
    def trie_insert(self, addr, value):
        copied_inner = copy.copy(self.inner)
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if first not in copied_inner:
                submap = Trie(hashable_dict())
            else:
                submap = copied_inner[first]
            new_submap = submap.trie_insert(rest, value)
            copied_inner[first] = new_submap
            return Trie(copied_inner)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            copied_inner[addr] = value
            return Trie(copied_inner)

    def has_submap(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_submap(first):
                submap = self.get_submap(first)
                return submap.has_submap(rest)
            else:
                return False
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return addr in self.inner

    def get_submap(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_submap(first):
                submap = self.get_submap(first)
                return submap.get_submap(rest)
            else:
                return None
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            if addr not in self.inner:
                return None
            return self.inner[addr]

    def get_submaps_shallow(self):
        return self.inner.items()

    ###########
    # Dunders #
    ###########

    def __getitem__(self, k):
        return self.get_submap(k)

    def __contains__(self, k):
        return self.has_submap(k)

    def __hash__(self):
        return hash(self.inner)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        for k, v in self.get_submaps_shallow():
            subk = tree.add(f"[bold]:{k}")
            _ = v.__rich_tree__(subk)
        return tree

    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        for k, v in self.inner.items():
            subk = tree.add(f"[bold]:{k}")
            submap = gpp._pformat(v, **kwargs)
            subk.add(submap)
        return tree
