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

"""
This module holds an abstract class which forms the core functionality
for "tree-like" classes (like :code:`ChoiceMap` and :code:`Selection`).
"""

import abc
from dataclasses import dataclass

from rich.tree import Tree

import genjax.core.pretty_printing as gpp
from genjax.core.pytree import Pytree


@dataclass
class ChoiceTree(Pytree):
    @abc.abstractmethod
    def is_leaf(self):
        pass

    @abc.abstractmethod
    def get_leaf_value(self):
        pass

    @abc.abstractmethod
    def has_subtree(self, addr):
        pass

    @abc.abstractmethod
    def get_subtree(self, addr):
        pass

    @abc.abstractmethod
    def get_subtrees_shallow(self):
        pass

    @abc.abstractmethod
    def merge(self, other):
        pass

    def tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        for (k, v) in self.get_subtrees_shallow():
            subk = tree.add(f"[bold][green]{k}")
            if hasattr(v, "build_rich_tree"):
                subt = v.build_rich_tree()
                subk.add(subt)
            else:
                subk.add(gpp.tree_pformat(v))

        return tree
