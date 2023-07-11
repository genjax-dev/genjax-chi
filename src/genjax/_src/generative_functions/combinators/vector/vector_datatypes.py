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
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck


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
    indices: IntArray
    inner: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.indices,
            self.inner,
            self.args,
            self.retval,
            self.score,
        ), ()

    def get_args(self):
        return self.args

    def get_choices(self):
        return VectorChoiceMap.new(self.indices, self.inner)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def project(self, selection: Selection) -> FloatArray:
        if isinstance(selection, VectorSelection):
            return self.inner.project(selection.inner)[selection.indices]
        if isinstance(selection, AllSelection):
            return self.score
        else:
            return 0.0


#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    indices: IntArray
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.indices, self.inner), ()

    @classmethod
    def _static_check_broadcast_dim_length(cls, tree):
        broadcast_dim_tree = jtu.tree_map(lambda v: len(v), tree)
        leaves = jtu.tree_leaves(broadcast_dim_tree)
        leaf_lengths = set(leaves)
        # all the leaves must have the same first dim size.
        assert len(leaf_lengths) == 1
        max_index = list(leaf_lengths).pop()
        return max_index

    @typecheck
    @classmethod
    def new(cls, indices, inner: ChoiceMap) -> ChoiceMap:
        # if you try to wrap around an EmptyChoiceMap, do nothing.
        if isinstance(inner, EmptyChoiceMap):
            return inner

        # convert list to array.
        if isinstance(indices, list):
            # indices can't be empty.
            assert indices
            indices = jnp.array(indices)
        indices_len = len(indices)
        inner_len = VectorChoiceMap._static_check_broadcast_dim_length(inner)
        # indices must have same length as leaves of the inner choice map.
        assert indices_len == inner_len

        return VectorChoiceMap(indices, inner)

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
            return VectorChoiceMap(other.indices, self.inner.merge(other.inner))
        else:
            return VectorChoiceMap(self.indices, self.inner.merge(other))

    def __hash__(self):
        return hash(self.inner)

    def get_index(self):
        return self.indices

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
    indices: Any
    inner: Selection

    def flatten(self):
        return (self.inner,), (self.indices,)

    @classmethod
    def new(cls, indices, inner):
        return VectorSelection(indices, inner)

    def filter(self, tree):
        assert isinstance(tree, VectorChoiceMap) or isinstance(tree, VectorTrace)
        filtered = self.inner.filter(tree)
        return VectorChoiceMap(tree.indices, filtered)

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
    index: Int
    inner: ChoiceMap

    def flatten(self):
        return (self.index, self.inner), ()

    @typecheck
    @classmethod
    def new(cls, index, inner: ChoiceMap) -> ChoiceMap:
        # if you try to wrap around an EmptyChoiceMap, do nothing.
        if isinstance(inner, EmptyChoiceMap):
            return inner

        return IndexChoiceMap(index, inner)

    def get_selection(self):
        subselection = self.inner.get_selection()
        return IndexSelection.new(subselection)

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
            return VectorChoiceMap(other.indices, self.inner.merge(other.inner))
        else:
            return VectorChoiceMap(self.indices, self.inner.merge(other))

    def __hash__(self):
        return hash(self.inner)

    def get_index(self):
        return self.indices

    def _tree_console_overload(self):
        tree = Tree(f"[b]{self.__class__.__name__}[/b]")
        subt = self.inner._build_rich_tree()
        subk = Tree("[blue]indices")
        subk.add(gpp.tree_pformat(self.indices))
        tree.add(subk)
        tree.add(subt)
        return tree


#####
# IndexSelection
#####


@dataclass
class IndexSelection(Selection):
    index: Int
    inner: Selection

    def flatten(self):
        return (
            self.index,
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
