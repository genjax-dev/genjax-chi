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
from enum import Enum
from typing import Any
from typing import Tuple

import jax
import jax.core as jc
import jax.numpy as jnp
import rich

from genjax._src.core.datatypes.address_tree import AddressLeaf
from genjax._src.core.datatypes.address_tree import AddressTree
from genjax._src.core.pretty_printing import CustomPretty
from genjax._src.core.typing import Bool
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import static_check_is_array


@dataclass
class TraceType(AddressTree):
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


BaseMeasure = Enum("BaseMeasure", ["Counting", "Lebesgue", "Bottom"])


@dataclass
class AddressLeafTraceType(TraceType, AddressLeaf):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    @abc.abstractmethod
    def get_base_measure(self) -> BaseMeasure:
        pass

    @abc.abstractmethod
    def check_subset(self, other) -> Bool:
        pass

    def check_shape(self, other) -> Bool:
        return self.shape == other.shape

    def check_base_measure(self, other) -> Bool:
        m1 = self.get_base_measure()
        m2 = other.get_base_measure()
        return m1 == m2

    def __check__(self, other) -> Bool:
        shape_check = self.check_shape(other)
        measure_check = self.check_base_measure(other)
        subset_check = self.check_subset(other)
        check = (
            (shape_check and measure_check and subset_check)
            or isinstance(other, Bottom)
            or isinstance(self, Bottom)
        )
        return check

    def on_support(self, other):
        assert isinstance(other, TraceType)
        check = self.__check__(other)
        if check:
            return check, ()
        else:
            return check, (self, other)

    def get_leaf_value(self):
        raise Exception("AddressLeafTraceType doesn't keep a leaf value.")

    def set_leaf_value(self):
        raise Exception("AddressLeafTraceType doesn't allow setting a leaf value.")

    def get_rettype(self):
        return self


@dataclass
class Reals(AddressLeafTraceType, CustomPretty):
    def get_base_measure(self) -> BaseMeasure:
        return BaseMeasure.Lebesgue

    def check_subset(self, other):
        return isinstance(other, Reals) or isinstance(other, Bottom)

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℝ[/b] {self.shape}")
        return tree


@dataclass
class PositiveReals(AddressLeafTraceType, CustomPretty):
    def get_base_measure(self):
        return BaseMeasure.Lebesgue

    def check_subset(self, other):
        return (
            isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℝ⁺[/b] {self.shape}")
        return tree


@dataclass
class RealInterval(AddressLeafTraceType, CustomPretty):
    lower_bound: Any
    upper_bound: Any

    def flatten(self):
        return (), (self.shape, self.lower_bound, self.upper_bound)

    def get_base_measure(self):
        return BaseMeasure.Lebesgue

    # TODO: we need to check if `lower_bound` and `upper_bound`
    # are concrete.
    def check_subset(self, other):
        return (
            isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_free(self, **kwargs):
        tree = rich.tree.Tree(
            f"[b]ℝ[/b] [{self.lower_bound}, {self.upper_bound}]{self.shape}"
        )
        return tree


@dataclass
class Integers(AddressLeafTraceType, CustomPretty):
    def flatten(self):
        return (), (self.shape,)

    def get_base_measure(self):
        return BaseMeasure.Counting

    def check_subset(self, other):
        return (
            isinstance(other, Integers)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℤ[/b] {self.shape}")
        return tree


@dataclass
class Naturals(AddressLeafTraceType, CustomPretty):
    def get_base_measure(self):
        return BaseMeasure.Counting

    def subset_check(self, other):
        return (
            isinstance(other, Naturals)
            or isinstance(other, Integers)
            or isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]ℕ[/b] {self.shape}")
        return tree


@dataclass
class Finite(AddressLeafTraceType, CustomPretty):
    limit: IntArray

    def get_base_measure(self):
        return BaseMeasure.Counting

    def check_subset(self, other):
        return (
            isinstance(other, Naturals)
            or isinstance(other, Integers)
            or isinstance(other, PositiveReals)
            or isinstance(other, Reals)
            or isinstance(other, Bottom)
        )

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree(f"[b]𝔽[/b] [{self.limit}] {self.shape}")
        return tree


@dataclass
class Bottom(AddressLeafTraceType, CustomPretty):
    def __init__(self):
        super().__init__(())

    def get_base_measure(self):
        return BaseMeasure.Bottom

    def check_subset(self, other):
        return True

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree("[b]⊥[/b]")
        return tree


@dataclass
class Empty(AddressLeafTraceType, CustomPretty):
    def __init__(self):
        super().__init__(())

    def check_subset(self, other):
        return True

    # Pretty sure this is wrong - but `Empty` can't occur
    # in distributions return anyways.
    def get_base_measure(self):
        return BaseMeasure.Counting

    # CustomPretty.
    def pformat_tree(self, **kwargs):
        tree = rich.tree.Tree("[b]ϕ[/b] (empty)")
        return tree


# Lift Python values to the trace type lattice.
def tt_lift(v, shape=()):
    if v is None:
        return Empty()
    elif v == jnp.int32:
        return Integers(shape)
    elif v == jnp.float32:
        return Reals(shape)
    elif v == bool:
        return Finite(shape, 2)
    elif static_check_is_array(v):
        return tt_lift(v.dtype, shape=v.shape)
    elif isinstance(v, jax.ShapeDtypeStruct):
        return tt_lift(v.dtype, shape=v.shape)
    elif isinstance(v, jc.ShapedArray):
        return tt_lift(v.dtype, shape=v.shape)
