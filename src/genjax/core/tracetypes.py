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

import abc
from dataclasses import dataclass
from typing import Any
from typing import Tuple

import numpy as np
from rich.tree import Tree

from genjax.core.choice_tree import ChoiceTree


@dataclass
class TraceType(ChoiceTree):
    def subseteq(self, other):
        assert isinstance(other, TraceType)
        check = self.__subseteq__(other)
        if check:
            return check, ()
        else:
            return check, (self, other)

    @abc.abstractmethod
    def __subseteq__(self, other):
        pass

    @abc.abstractmethod
    def get_rettype(self):
        pass


@dataclass
class LeafTraceType(TraceType):
    def is_leaf(self):
        return True

    def get_leaf_value(self):
        return self

    def has_subtree(self, addr):
        return False

    @classmethod
    def get_subtree(cls, addr):
        raise Exception(f"{cls} is a leaf choice tree.")

    def get_subtrees_shallow(self):
        return ()

    def merge(self, other):
        return other


@dataclass
class Reals(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __subseteq__(self, other):
        if isinstance(other, Reals):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_types_shallow(self):
        return ()

    def get_rettype(self):
        return self

    def tree_console_overload(self):
        tree = Tree(f"[magenta][b]ℝ[/b] {self.shape}")
        return tree


@dataclass
class PositiveReals(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __subseteq__(self, other):
        if isinstance(other, PositiveReals):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_types_shallow(self):
        return ()

    def get_rettype(self):
        return self

    def tree_console_overload(self):
        tree = Tree(f"[magenta][b]ℝ⁺[/b] {self.shape}")
        return tree


@dataclass
class RealInterval(LeafTraceType):
    shape: Tuple
    lower_bound: Any
    upper_bound: Any

    def flatten(self):
        return (), (self.shape, self.lower_bound, self.upper_bound)

    def __subseteq__(self, other):
        if isinstance(other, Reals):
            return np.sum(self.shape) <= np.sum(other.shape)
        elif isinstance(other, PositiveReals):
            return self.lower_bound >= 0.0 and np.sum(self.shape) <= np.sum(
                other.shape
            )
        else:
            return False

    def get_types_shallow(self):
        return ()

    def get_rettype(self):
        return self

    def tree_console_overload(self):
        tree = Tree(
            f"[magenta][b]ℝ[/b] [{self.lower_bound}, {self.upper_bound}]{self.shape}"
        )
        return tree


@dataclass
class Integers(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __subseteq__(self, other):
        if isinstance(other, Integers) or isinstance(other, Reals):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_types_shallow(self):
        return ()

    def get_rettype(self):
        return self

    def tree_console_overload(self):
        tree = Tree(f"[magenta][b]ℤ[/b] {self.shape}")
        return tree


@dataclass
class Naturals(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __subseteq__(self, other):
        if (
            isinstance(other, Naturals)
            or isinstance(other, Reals)
            or isinstance(other, PositiveReals)
        ):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_types_shallow(self):
        return ()

    def get_rettype(self):
        return self

    def tree_console_overload(self):
        tree = Tree(f"[magenta][b]ℕ[/b] {self.shape}")
        return tree


@dataclass
class Finite(LeafTraceType):
    shape: Tuple
    limit: int

    def flatten(self):
        return (), (self.shape, self.limit)

    def __subseteq__(self, other):
        if (
            isinstance(other, Naturals)
            or isinstance(other, Reals)
            or isinstance(other, PositiveReals)
        ):
            return np.sum(self.shape) <= np.sum(other.shape)
        elif isinstance(other, Finite):
            return self.limit <= other.limit and np.sum(self.shape) <= np.sum(
                other.shape
            )
        else:
            return False

    def get_types_shallow(self):
        return ()

    def get_rettype(self):
        return self

    def tree_console_overload(self):
        tree = Tree(f"[magenta][b]𝔽[/b] [{self.limit}] {self.shape}")
        return tree


@dataclass
class Bottom(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __subseteq__(self, other):
        return np.sum(self.shape) <= np.sum(other.shape)

    def get_types_shallow(self):
        return ()

    def get_rettype(self):
        return self

    def tree_console_overload(self):
        tree = Tree("[magenta][b]⊥[/b]")
        return tree
