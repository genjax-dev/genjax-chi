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
from genjax._src.core.datatypes.generative import EmptyChoice
from genjax._src.core.datatypes.generative import IndexedChoiceMap
from genjax._src.core.datatypes.generative import IndexedSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import choice_map
from genjax._src.core.pytree.checks import (
    static_check_tree_leaves_have_matching_leading_dim,
)
from genjax._src.core.typing import Dict
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck


######################################
# Vector-shaped combinator datatypes #
######################################

# The data types in this section are used in `Map` and `Unfold`, currently.


#####################
# Vector choice map #
#####################


@dataclass
class VectorChoiceMap(ChoiceMap):
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    @dispatch
    def new(
        cls,
        inner: EmptyChoice,
    ) -> EmptyChoice:
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

    @typecheck
    def filter(
        self,
        selection: Selection,
    ) -> ChoiceMap:
        return VectorChoiceMap.new(self.inner.filter(selection))

    def get_selection(self):
        subselection = self.inner.get_selection()
        # Static: get the leading dimension size value.
        dim = static_check_tree_leaves_have_matching_leading_dim(
            self.inner,
        )
        return IndexedSelection(jnp.arange(dim), subselection)

    def has_submap(self, addr):
        return self.inner.has_submap(addr)

    def get_submap(self, addr):
        return self.inner.get_submap(addr)

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

        assert jtu.tree_structure(self.inner) == jtu.tree_structure(new)
        new = jtu.tree_map(_inner, self.inner, new)

        return VectorChoiceMap(new), IndexedChoiceMap(indices, discard)

    @dispatch
    def merge(self, other: EmptyChoice) -> Tuple[ChoiceMap, ChoiceMap]:
        return self, other

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        sub_tree = rich.tree.Tree("[bold](Vector)")
        self.inner.__rich_tree__(sub_tree)
        tree.add(sub_tree)
        return tree


##############
# Shorthands #
##############

vector_choice_map = VectorChoiceMap.new
