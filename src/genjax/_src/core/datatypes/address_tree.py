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
"tree-like" classes (like `ChoiceMap` and `Selection`)."""

import abc
from dataclasses import dataclass

from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import BoolArray


@dataclass
class AddressMap(Pytree):
    """> The `AddressMap` class is used to define abstract classes for tree-
    shaped datatypes. These classes are used to implement trace, choice map,
    and selection types.

    One should think of `AddressMap` as providing a convenient base class for
    many of the generative datatypes declared in GenJAX. `AddressMap` mixes in
    `Pytree` automatically.
    """

    @abc.abstractmethod
    def has_submap(self, addr) -> BoolArray:
        pass

    @abc.abstractmethod
    def get_submap(self, addr):
        pass


@dataclass
class AddressLeaf(AddressMap):
    """> The `AddressLeaf` class specializes `AddressMap` to classes without
    any internal submaps.

    `AddressLeaf` is a convenient base for generative datatypes which don't keep reference to other `AddressMap` instances - things like `ValueChoiceMap` (whose only choice value is a single value, not a dictionary or other tree-like object). `AddressLeaf` extends `AddressMap` with a special extension method `get_leaf_value`.
    """

    @abc.abstractmethod
    def get_leaf_value(self):
        pass

    @abc.abstractmethod
    def set_leaf_value(self, v):
        pass

    def has_submap(self, addr):
        return False

    def get_submap(self, addr):
        raise Exception(
            f"{type(self)} is a AddressLeaf: it does not address any internal choices."
        )
