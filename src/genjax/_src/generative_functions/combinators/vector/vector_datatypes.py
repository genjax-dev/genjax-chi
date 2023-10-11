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

import jax.numpy as jnp
import jax.tree_util as jtu
import rich

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import ComplementIndexedSelection
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import IndexedChoiceMap
from genjax._src.core.datatypes.generative import IndexedSelection
from genjax._src.core.datatypes.generative import Mask
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import TraceType
from genjax._src.core.datatypes.generative import choice_map
from genjax._src.core.datatypes.generative import mask
from genjax._src.core.pytree.static_checks import (
    static_check_tree_leaves_have_matching_leading_dim,
)
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch


######################################
# Vector-shaped combinator datatypes #
######################################

# The data types in this section are used in `Map` and `Unfold`, currently.


#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    @dispatch
    def new(
        cls,
        inner: EmptyChoiceMap,
    ) -> EmptyChoiceMap:
        return inner

    @classmethod
    @dispatch
    def new(
        cls,
        inner: ChoiceMap,
    ) -> ChoiceMap:
        # Static assertion: all leaves must have same first dim size.
        static_check_tree_leaves_have_matching_leading_dim(inner)
        return VectorChoiceMap(inner)

    @classmethod
    @dispatch
    def new(
        cls,
        inner: Dict,
    ) -> ChoiceMap:
        chm = choice_map(inner)
        return VectorChoiceMap.new(chm)

    def is_empty(self):
        return self.inner.is_empty()

    @dispatch
    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        return VectorChoiceMap.new(self.inner.filter(selection))

    @dispatch
    def filter(
        self,
        selection: IndexedSelection,
    ) -> Mask:
        filtered = self.inner.filter(selection.inner)
        flags = jnp.logical_and(
            selection.indices >= 0,
            selection.indices
            < static_check_tree_leaves_have_matching_leading_dim(self.inner),
        )

        def _take(v):
            return jnp.take(v, selection.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    @dispatch
    def filter(
        self,
        selection: ComplementIndexedSelection,
    ) -> Mask:
        filtered = self.inner.filter(selection.inner.complement())
        flags = jnp.logical_not(
            jnp.logical_and(
                selection.indices >= 0,
                selection.indices
                < static_check_tree_leaves_have_matching_leading_dim(self.inner),
            )
        )

        def _take(v):
            return jnp.take(v, selection.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    def get_selection(self):
        subselection = self.inner.get_selection()
        return subselection

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return self.inner.get_subtree(addr)

    def get_subtrees_shallow(self):
        return self.inner.get_subtrees_shallow()

    @dispatch
    def merge(self, other: "VectorChoiceMap") -> Tuple[ChoiceMap, ChoiceMap]:
        new, discard = self.inner.merge(other.inner)
        return VectorChoiceMap(new), VectorChoiceMap(discard)

    @dispatch
    def merge(self, other: IndexedChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        indices = other.indices

        sliced = jtu.tree_map(lambda v: v[indices], self.inner)
        new, discard = sliced.merge(other.inner)

        def _inner(v1, v2):
            return v1.at[indices].set(v2)

        print(jtu.tree_structure(self.inner))
        print()
        print(jtu.tree_structure(new))
        assert jtu.tree_structure(self.inner) == jtu.tree_structure(new)
        new = jtu.tree_map(_inner, self.inner, new)

        return VectorChoiceMap(new), IndexedChoiceMap(indices, discard)

    @dispatch
    def merge(self, other: EmptyChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        return self, other

    ###########
    # Dunders #
    ###########

    def __hash__(self):
        return hash(self.inner)

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Vector)")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


#####
# VectorTraceType
#####


@dataclass
class VectorTraceType(TraceType):
    inner: TraceType
    length: int

    def flatten(self):
        return (), (self.inner, self.length)

    def is_leaf(self):
        return self.inner.is_leaf()

    def get_leaf_value(self):
        return self.inner.get_leaf_value()

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        v = self.inner.get_subtree(addr)
        return VectorTraceType(v, self.length)

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return (k, VectorTraceType(v, self.length))

        return map(lambda args: _inner(*args), self.inner.get_subtrees_shallow())

    def merge(self, _):
        raise Exception("Not implemented.")

    def __subseteq__(self, _):
        return False

    def get_rettype(self):
        return self.inner.get_rettype()


##############
# Shorthands #
##############

vector_choice_map = VectorChoiceMap.new
