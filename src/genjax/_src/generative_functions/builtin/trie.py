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

import rich

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.hashabledict import HashableDict
from genjax._src.core.hashabledict import hashabledict
from genjax._src.core.tree import Tree


#####
# Trie
#####


@dataclass
class Trie(Tree):
    inner: HashableDict

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    def new(cls):
        return Trie(hashabledict())

    def trie_insert(self, addr, value):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if first not in self.inner:
                subtree = Trie(hashabledict())
                self.inner[first] = subtree
            subtree = self.inner[first]
            subtree.trie_insert(rest, value)
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            self.inner[addr] = value

    def has_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.has_subtree(rest)
            else:
                return False
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            return addr in self.inner

    def get_subtree(self, addr):
        if isinstance(addr, tuple) and len(addr) > 1:
            first, *rest = addr
            rest = tuple(rest)
            if self.has_subtree(first):
                subtree = self.get_subtree(first)
                return subtree.get_subtree(rest)
            else:
                return None
        else:
            if isinstance(addr, tuple):
                addr = addr[0]
            if addr not in self.inner:
                return None
            return self.inner[addr]

    def get_subtrees_shallow(self):
        return self.inner.items()

    def merge(self, other):
        new = hashabledict()
        for (k, v) in self.get_subtrees_shallow():
            if other.has_subtree(k):
                sub = other.get_subtree(k)
                new[k] = v.merge(sub)
            else:
                new[k] = v
        for (k, v) in other.get_subtrees_shallow():
            if not self.has_subtree(k):
                new[k] = v
        return Trie(new)

    def __setitem__(self, k, v):
        self.trie_insert(k, v)

    def __getitem__(self, k):
        return self.get_subtree(k)

    def __contains__(self, k):
        return self.has_subtree(k)

    def __hash__(self):
        return hash(self.inner)

    def _pformat_custom(self, **kwargs):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        for (k, v) in self.inner.items():
            subk = tree.add(f"[bold green]{k}")
            subtree = gpp._pformat(v, **kwargs)
            subk.add(subtree)
        return tree
