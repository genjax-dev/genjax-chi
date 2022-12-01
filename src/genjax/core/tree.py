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

"""This module holds an abstract class which forms the core functionality for
"tree-like" classes (like :code:`ChoiceMap` and :code:`Selection`)."""

import abc
import dataclasses
from dataclasses import dataclass

import rich

import genjax.core.pretty_printing as gpp
from genjax.core.pytree import Pytree


@dataclass
class Tree(Pytree):
    @classmethod
    def new(cls, *args):
        return cls(*args)

    @abc.abstractmethod
    def has_subtree(self, addr) -> bool:
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

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        for (k, v) in self.get_subtrees_shallow():
            subk = tree.add(f"[bold][green]{k}")
            if hasattr(v, "_build_rich_tree"):
                subt = v._build_rich_tree()
                subk.add(subt)
            else:
                subk.add(gpp.tree_pformat(v))

        return tree


@dataclass
class Leaf(Tree):
    @abc.abstractmethod
    def get_leaf_value(self):
        pass

    def has_subtree(self, addr):
        return False

    def get_subtree(self, addr):
        raise Exception(
            f"{type(self)} is a Leaf: it does not address any internal choices."
        )

    def get_subtrees_shallow(self):
        return ()

    def merge(self, other):
        return other

    def _build_rich_tree(self):
        tree = rich.tree.Tree(f"[b]{self.__class__.__name__}[/b]")
        if dataclasses.is_dataclass(self):
            d = dict(
                (field.name, getattr(self, field.name))
                for field in dataclasses.fields(self)
            )
            for (k, v) in d.items():
                subk = tree.add(f"[blue]{k}")
                if isinstance(v, Pytree) or hasattr(v, "_build_rich_tree"):
                    subtree = v._build_rich_tree()
                    subk.add(subtree)
                else:
                    subk.add(gpp.tree_pformat(v))
        return tree
