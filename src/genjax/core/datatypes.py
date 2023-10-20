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

from genjax._src.core.datatypes.address_tree import AddressLeaf
from genjax._src.core.datatypes.address_tree import AddressTree
from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import Bottom
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import ComplementHierarchicalSelection
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import Finite
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalChoiceMap
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import IndexedChoiceMap
from genjax._src.core.datatypes.generative import IndexedSelection
from genjax._src.core.datatypes.generative import Integers
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import Mask
from genjax._src.core.datatypes.generative import Naturals
from genjax._src.core.datatypes.generative import NoneSelection
from genjax._src.core.datatypes.generative import PositiveReals
from genjax._src.core.datatypes.generative import RealInterval
from genjax._src.core.datatypes.generative import Reals
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import TaggedUnion
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.datatypes.generative import all_select
from genjax._src.core.datatypes.generative import choice_map
from genjax._src.core.datatypes.generative import empty_choice_map
from genjax._src.core.datatypes.generative import indexed_choice_map
from genjax._src.core.datatypes.generative import indexed_select
from genjax._src.core.datatypes.generative import mask
from genjax._src.core.datatypes.generative import none_select
from genjax._src.core.datatypes.generative import select
from genjax._src.core.datatypes.generative import tagged_union
from genjax._src.core.datatypes.generative import value_choice_map
from genjax._src.core.datatypes.hashable_dict import HashableDict
from genjax._src.core.datatypes.hashable_dict import hashable_dict
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.datatypes.trie import trie
from genjax._src.core.pytree import DynamicClosure
from genjax._src.core.pytree import Pytree
from genjax._src.core.pytree import dynamic_closure


__all__ = [
    # Hashable dictionary type.
    "HashableDict",
    "hashable_dict",
    # Trie type.
    "Trie",
    "trie",
    # Generative datatypes.
    "ChoiceMap",
    "EmptyChoiceMap",
    "empty_choice_map",
    "ValueChoiceMap",
    "value_choice_map",
    "HierarchicalChoiceMap",
    "choice_map",
    "IndexedChoiceMap",
    "indexed_choice_map",
    "Trace",
    "Selection",
    "AllSelection",
    "all_select",
    "NoneSelection",
    "none_select",
    "HierarchicalSelection",
    "select",
    "IndexedSelection",
    "indexed_select",
    "ComplementHierarchicalSelection",
    "GenerativeFunction",
    "JAXGenerativeFunction",
    # Masking.
    "Mask",
    "mask",
    # Tagged unions.
    "TaggedUnion",
    "tagged_union",
    # Trace types.
    "Bottom",
    "Reals",
    "PositiveReals",
    "RealInterval",
    "Integers",
    "Naturals",
    "Finite",
    "Bottom",
    # Pytree meta.
    "Pytree",
    "DynamicClosure",
    "dynamic_closure",
    "AddressTree",
    "AddressLeaf",
]
