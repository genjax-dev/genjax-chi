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

import jax
import jax.tree_util as jtu

from genjax._src.core.datatypes.address_tree import AddressLeaf
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import typecheck


@dataclass
class TaggedUnion(Pytree):
    tag: IntArray
    values: List[Any]

    def flatten(self):
        return (self.tag, self.values), ()

    @classmethod
    def new(cls, tag: IntArray, values: List[Any]):
        return cls(tag, values).leaf_push()

    def _set_leafs(self, *vs: AddressLeaf):
        leaf_values = list(map(lambda v: v.get_leaf_value(), vs))
        first = vs[0]
        return first.set_leaf_value(TaggedUnion(self.tag, leaf_values))

    def leaf_push(self):
        def _inner(*v):
            # `AddressLeaf` inheritors have a method `set_leaf_value`
            # to participate in tagging.
            # They can choose how to construct themselves after
            # being provided with a tagged value.
            if all(map(lambda v: isinstance(v, AddressLeaf), v)):
                return self._set_leafs(*v)
            else:
                return list(v)

        def _check(v):
            return isinstance(v, AddressLeaf)

        return jtu.tree_map(_inner, *self.values, is_leaf=_check)

    @typecheck
    def switch(self, *callables: Callable):
        return jax.lax.switch(self.tag, *callables)


##############
# Shorthands #
##############

tagged_union = TaggedUnion.new
