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
from dataclasses import dataclass

from genjax._src.core.pytree import Pytree


@dataclass
class Tree(Pytree):
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


@dataclass
class Leaf(Tree):
    @abc.abstractmethod
    def get_leaf_value(self):
        pass

    @abc.abstractmethod
    def set_leaf_value(self, v):
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
