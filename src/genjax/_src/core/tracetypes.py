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

import rich

from genjax._src.core.tree import Leaf
from genjax._src.core.tree import Tree


@dataclass
class TraceType(Tree):
    def on_support(self, other):
        assert isinstance(other, TraceType)
        check = self.__check__(other)
        if check:
            return check, ()
        else:
            return check, (self, other)

    @abc.abstractmethod
    def __check__(self, other):
        pass

    @abc.abstractmethod
    def get_rettype(self):
        pass

    # TODO: think about this.
    # Overload now to play nicely with `Selection`.
    def get_choices(self):
        return self

    def __getitem__(self, addr):
        sub = self.get_subtree(addr)
        return sub


@dataclass
class LeafTraceType(TraceType, Leaf):
    def get_leaf_value(self):
        raise Exception("LeafTraceType doesn't keep a leaf value.")

    def set_leaf_value(self):
        raise Exception("LeafTraceType doesn't allow setting a leaf value.")

    def get_rettype(self):
        return self


@dataclass
class Reals(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __check__(self, other):
        if isinstance(other, Reals):
            return self.shape == other.shape
        elif isinstance(other, Bottom):
            return True
        else:
            return False

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[magenta][b]ℝ[/b] {self.shape}")
        return tree


@dataclass
class PositiveReals(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __check__(self, other):
        if isinstance(other, PositiveReals):
            return self.shape == other.shape
        elif isinstance(other, Bottom):
            return True
        else:
            return False

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[magenta][b]ℝ⁺[/b] {self.shape}")
        return tree


@dataclass
class RealInterval(TraceType, Leaf):
    shape: Tuple
    lower_bound: Any
    upper_bound: Any

    def flatten(self):
        return (), (self.shape, self.lower_bound, self.upper_bound)

    def __check__(self, other):
        if isinstance(other, PositiveReals):
            return self.lower_bound >= 0.0 and self.shape == other.shape
        elif isinstance(other, Bottom):
            return True
        else:
            return False

    def _tree_console_overload(self):
        tree = rich.tree.Tree(
            f"[magenta][b]ℝ[/b] [{self.lower_bound}, {self.upper_bound}]{self.shape}"
        )
        return tree


@dataclass
class Integers(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __check__(self, other):
        if isinstance(other, Integers) or isinstance(other, Reals):
            return self.shape == other.shape
        elif isinstance(other, Bottom):
            return True
        else:
            return False

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[magenta][b]ℤ[/b] {self.shape}")
        return tree


@dataclass
class Naturals(LeafTraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    def __check__(self, other):
        if (
            isinstance(other, Naturals)
            or isinstance(other, Reals)
            or isinstance(other, PositiveReals)
        ):
            return self.shape <= other.shape
        elif isinstance(other, Bottom):
            return True
        else:
            return False

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[magenta][b]ℕ[/b] {self.shape}")
        return tree


@dataclass
class Finite(TraceType, Leaf):
    shape: Tuple
    limit: int

    def flatten(self):
        return (), (self.shape, self.limit)

    def __check__(self, other):
        if (
            isinstance(other, Naturals)
            or isinstance(other, Reals)
            or isinstance(other, PositiveReals)
        ):
            return self.shape <= other.shape
        elif isinstance(other, Finite):
            return self.limit <= other.limit and self.shape <= other.shape
        elif isinstance(other, Bottom):
            return True
        else:
            return False

    def _tree_console_overload(self):
        tree = rich.tree.Tree(f"[magenta][b]𝔽[/b] [{self.limit}] {self.shape}")
        return tree


@dataclass
class Bottom(LeafTraceType):
    def flatten(self):
        return (), ()

    def __check__(self, other):
        return True

    def _tree_console_overload(self):
        tree = rich.tree.Tree("[magenta][b]⊥[/b]")
        return tree
