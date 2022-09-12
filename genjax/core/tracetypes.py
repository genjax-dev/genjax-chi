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
import numpy as np
import jax._src.pretty_printer as pp
from genjax.core.pytree import Pytree
from typing import Tuple, Any


@dataclass
class TraceType(Pytree):
    def subseteq(self, other):
        assert isinstance(other, TraceType)
        check = self.__subseteq__(other)
        if check:
            return check, ()
        else:
            return check, (self, other)

    def overload_pprint(self, **kwargs):
        return pp.text(self.__repr__())

    @abc.abstractmethod
    def __subseteq__(self, other):
        pass

    @abc.abstractmethod
    def get_rettype(self):
        pass


@dataclass
class Reals(TraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    @classmethod
    def unflatten(cls, xs, data):
        return Reals(*xs, *data)

    def __subseteq__(self, other):
        if isinstance(other, Reals):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_rettype(self):
        return self

    def __repr__(self):
        return f"ℝ (shape = {self.shape})"

    def __str__(self):
        return f"ℝ (shape = {self.shape})"


@dataclass
class PositiveReals(TraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    @classmethod
    def unflatten(cls, xs, data):
        return PositiveReals(*xs, *data)

    def __subseteq__(self, other):
        if isinstance(other, PositiveReals):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_rettype(self):
        return self

    def __repr__(self):
        return f"ℝ⁺ (shape = {self.shape})"

    def __str__(self):
        return f"ℝ⁺ (shape = {self.shape})"


@dataclass
class RealInterval(TraceType):
    shape: Tuple
    lower_bound: Any
    upper_bound: Any

    def flatten(self):
        return (), (self.shape, self.lower_bound, self.upper_bound)

    @classmethod
    def unflatten(cls, xs, data):
        return RealInterval(*xs, *data)

    def __subseteq__(self, other):
        if isinstance(other, Reals):
            return np.sum(self.shape) <= np.sum(other.shape)
        elif isinstance(other, PositiveReals):
            return self.lower_bound >= 0.0 and np.sum(self.shape) <= np.sum(
                other.shape
            )
        else:
            return False

    def get_rettype(self):
        return self

    def __repr__(self):
        return (
            f"ℝ[{self.lower_bound}, {self.upper_bound}] (shape = {self.shape})"
        )

    def __str__(self):
        return (
            f"ℝ[{self.lower_bound}, {self.upper_bound}] (shape = {self.shape})"
        )


@dataclass
class Integers(TraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    @classmethod
    def unflatten(cls, xs, data):
        return Integers(*xs, *data)

    def __subseteq__(self, other):
        if isinstance(other, Integers) or isinstance(other, Reals):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_rettype(self):
        return self

    def __repr__(self):
        return f"ℕ (shape = {self.shape})"

    def __str__(self):
        return f"ℕ (shape = {self.shape})"


@dataclass
class Naturals(TraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    @classmethod
    def unflatten(cls, xs, data):
        return Naturals(*xs, *data)

    def __subseteq__(self, other):
        if (
            isinstance(other, Naturals)
            or isinstance(other, Reals)
            or isinstance(other, PositiveReals)
        ):
            return np.sum(self.shape) <= np.sum(other.shape)
        else:
            return False

    def get_rettype(self):
        return self

    def __repr__(self):
        return f"ℕ (shape = {self.shape})"

    def __str__(self):
        return f"ℕ (shape = {self.shape})"


@dataclass
class Finite(TraceType):
    shape: Tuple
    limit: int

    def flatten(self):
        return (), (self.shape, self.limit)

    @classmethod
    def unflatten(cls, xs, data):
        return Finite(*xs, *data)

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

    def get_rettype(self):
        return self

    def __repr__(self):
        return f"𝔽[{self.limit}] (shape = {self.shape})"

    def __str__(self):
        return f"𝔽[{self.limit}] (shape = {self.shape})"


@dataclass
class Bottom(TraceType):
    shape: Tuple

    def flatten(self):
        return (), (self.shape,)

    @classmethod
    def unflatten(cls, xs, data):
        return Bottom(*xs, *data)

    def __subseteq__(self, other):
        return np.sum(self.shape) <= np.sum(other.shape)

    def get_rettype(self):
        return self

    def __repr__(self):
        return "⊥"

    def __str__(self):
        return "⊥"
