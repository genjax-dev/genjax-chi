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
from typing import Union

import jax.numpy as jnp
import jax.tree_util as jtu
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.typing import Any
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.vector.vector_utilities import (
    static_check_broadcast_dim_length,
)


######################################
# Vector-shaped combinator datatypes #
######################################

# The data types in this section are used in `Map` and `Unfold`, currently.

#####
# VectorTrace
#####


@dataclass
class VectorTrace(Trace):
    gen_fn: GenerativeFunction
    inner: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.inner,
            self.args,
            self.retval,
            self.score,
        ), ()

    def get_args(self):
        return self.args

    def get_choices(self):
        return VectorChoiceMap.new(self.inner)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def project(self, selection: Selection) -> FloatArray:
        if isinstance(selection, VectorSelection):
            if selection.masks is not None:
                return jnp.sum(
                    jnp.where(selection.masks, self.inner.project(selection.inner), 0.0)
                )
            else:
                return jnp.sum(self.inner.project(selection.inner))
        elif isinstance(selection, AllSelection):
            return self.score
        else:
            return 0.0


#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    masks: Union[None, BoolArray]
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.masks, self.inner), ()

    @typecheck
    @classmethod
    def new(
        cls, inner: ChoiceMap, masks: Union[None, list, BoolArray] = None
    ) -> ChoiceMap:
        # if you try to wrap around an EmptyChoiceMap, do nothing.
        if isinstance(inner, EmptyChoiceMap):
            return inner

        if isinstance(masks, list) or isinstance(masks, BoolArray):
            # indices can't be empty.
            assert masks
            masks = jnp.array(masks)
            masks_len = len(masks)
            inner_len = static_check_broadcast_dim_length(inner)
            # indices must have same length as leaves of the inner choice map.
            assert masks_len == inner_len
            return VectorChoiceMap(masks, inner)

        return VectorChoiceMap(None, inner)

    @typecheck
    @classmethod
    def convert(cls, choice_map: Trie):
        indices = []
        subtrees = []
        for (ind, subtree) in choice_map.get_subtrees_shallow():
            indices.append(ind)
            subtrees.append(subtree)

        # Assert that all Pytrees in list can be
        # stacked at leaves.
        # static_check_pytree_stackable(subtrees)
        # return VectorChoiceMap.new(indices, tree_stack(subtrees))

    def get_selection(self):
        subselection = self.inner.get_selection()
        return VectorSelection.new(subselection)

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return self.inner.get_subtree(addr)

    def get_subtrees_shallow(self):
        return self.inner.get_subtrees_shallow()

    # TODO: This currently provides poor support for merging
    # two vector choices maps with different index arrays.
    def merge(self, other):
        if isinstance(other, VectorChoiceMap):
            return VectorChoiceMap(other.masks, self.inner.merge(other.inner))
        else:
            return VectorChoiceMap(self.masks, self.inner.merge(other))

    def __hash__(self):
        return hash(self.inner)

    def _tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        subt = self.inner._build_rich_tree()
        subk = Tree("[blue]indices")
        subk.add(gpp.tree_pformat(self.indices))
        tree.add(subk)
        tree.add(subt)
        return tree


#####
# VectorSelection
#####


@dataclass
class VectorSelection(Selection):
    masks: Union[None, BoolArray]
    inner: Selection

    def flatten(self):
        return (self.masks, self.inner), ()

    @classmethod
    def new(cls, inner, masks=None):
        return VectorSelection(masks, inner)

    def filter(self, tree):
        assert isinstance(tree, VectorChoiceMap) or isinstance(tree, VectorTrace)
        filtered = self.inner.filter(tree)
        return VectorChoiceMap(tree.masks, filtered)

    def complement(self):
        return VectorSelection(self.inner.complement())

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return self.inner.get_subtree(addr)

    def get_subtrees_shallow(self):
        return self.inner.get_subtrees_shallow()

    def merge(self, other):
        assert isinstance(other, VectorSelection)
        return self.inner.merge(other)


#####
# IndexChoiceMap
#####


@dataclass
class IndexChoiceMap(ChoiceMap):
    indices: IntArray
    inner: ChoiceMap

    def flatten(self):
        return (self.indices, self.inner), ()

    @typecheck
    @classmethod
    def new(cls, inner: ChoiceMap, indices: Union[List, IntArray]) -> ChoiceMap:
        if isinstance(indices, List):
            indices = jnp.array(indices)

        # Verify that dimensions are consistent before creating an
        # `IndexChoiceMap`.
        _ = static_check_broadcast_dim_length((inner, indices))

        # if you try to wrap around an EmptyChoiceMap, do nothing.
        if isinstance(inner, EmptyChoiceMap):
            return inner

        return IndexChoiceMap(indices, inner)

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return jtu.tree_map(lambda v: v[self.indices], self.inner.get_subtree(addr))

    def get_selection(self):
        inner_selection = self.inner.get_selection()
        return IndexSelection.new(inner_selection, self.indices)

    def get_subtrees_shallow(self):
        pass


#####
# IndexSelection
#####


@dataclass
class IndexSelection(Selection):
    indices: IntArray
    inner: Selection

    def flatten(self):
        return (
            self.indices,
            self.inner,
        ), ()

    @classmethod
    def new(cls, indices, inner):
        pass

    def filter(self, tree):
        pass

    def complement(self):
        pass

    def has_subtree(self, addr):
        pass

    def get_subtree(self, addr):
        pass

    def get_subtrees_shallow(self):
        pass

    def merge(self, other):
        pass


##############
# Shorthands #
##############

vector_choice_map = VectorChoiceMap.new
vector_select = VectorSelection.new
index_choice_map = IndexChoiceMap.new
index_select = IndexSelection.new
