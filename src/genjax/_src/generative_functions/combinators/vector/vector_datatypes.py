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
from genjax._src.core.datatypes.generative import NoneSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.masks import mask
from genjax._src.core.pytree import tree_stack
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.vector.vector_utilities import (
    static_check_leaf_length,
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
            return jnp.sum(self.inner.project(selection.inner))
        elif isinstance(selection, IndexSelection) or isinstance(
            selection, ComplementIndexSelection
        ):
            inner_project = self.inner.project(selection.inner)
            return jnp.sum(
                jnp.take(inner_project, selection.indices, mode="fill", fill_value=0.0)
            )
        elif isinstance(selection, AllSelection):
            return self.score
        elif isinstance(selection, NoneSelection):
            return 0.0
        else:
            selection = VectorSelection.new(selection)
            return self.project(selection)


#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.inner,), ()

    @typecheck
    @classmethod
    def new(
        cls,
        inner: ChoiceMap,
    ) -> ChoiceMap:
        if isinstance(inner, EmptyChoiceMap):
            return inner
        return VectorChoiceMap(inner)

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
            return VectorChoiceMap(self.inner.merge(other.inner))
        else:
            return VectorChoiceMap(self.inner.merge(other))

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
    inner: Selection

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    def new(cls, inner):
        return VectorSelection(inner)

    def filter(self, tree):
        assert isinstance(tree, VectorChoiceMap) or isinstance(tree, VectorTrace)
        filtered = self.inner.filter(tree)
        return VectorChoiceMap(filtered)

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

    @classmethod
    def convert(cls, chm: ChoiceMap) -> "IndexChoiceMap":
        indices = []
        subtrees = []
        for (k, v) in chm.get_subtrees_shallow():
            if isinstance(k, IntArray):
                indices.append(k)
                subtrees.append(v)
            else:
                raise Exception(
                    f"Failed to convert choice map of type {type(chm)} to IndexChoiceMap."
                )

        inner = tree_stack(subtrees)
        indices = jnp.array(indices)
        return IndexChoiceMap.new(inner, indices)

    @typecheck
    @classmethod
    def new(cls, indices: Union[List, IntArray], inner: ChoiceMap) -> ChoiceMap:
        if isinstance(indices, List):
            indices = jnp.array(indices)

        # Verify that dimensions are consistent before creating an
        # `IndexChoiceMap`.
        _ = static_check_leaf_length((inner, indices))

        # if you try to wrap around an EmptyChoiceMap, do nothing.
        if isinstance(inner, EmptyChoiceMap):
            return inner

        return IndexChoiceMap(indices, inner)

    def get_index(self):
        return self.indices

    def has_subtree(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))

    @dispatch
    def get_subtree(self, addr: Tuple):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return EmptyChoiceMap()
        (idx, addr) = addr
        subtree = self.inner.get_subtree(addr)
        if isinstance(subtree, EmptyChoiceMap):
            return EmptyChoiceMap()
        else:
            (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
            subtree = jtu.tree_map(lambda v: v[slice_index], subtree)
            return mask(idx in self.indices, subtree)

    @dispatch
    def get_subtree(self, idx: IntArray):
        (slice_index,) = jnp.nonzero(idx == self.indices, size=1)
        slice_index = slice_index[0]
        subtree = jtu.tree_map(lambda v: v[slice_index] if v.shape else v, self.inner)
        return mask(jnp.isin(idx, self.indices), subtree)

    def get_selection(self):
        inner_selection = self.inner.get_selection()
        return IndexSelection.new(self.indices, inner_selection)

    def get_subtrees_shallow(self):
        raise NotImplementedError


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
    @dispatch
    def new(cls, idx: Union[Int, IntArray]):
        idxs = jnp.array(idx)
        return IndexSelection(idxs, AllSelection())

    @classmethod
    @dispatch
    def new(cls, idx: Union[Int, List[Int], IntArray], inner: Selection):
        idxs = jnp.array(idx)
        return IndexSelection(idxs, inner)

    def has_subtree(self, addr):
        if not isinstance(addr, Tuple) and len(addr) == 1:
            return False
        (idx, addr) = addr
        return jnp.logical_and(idx in self.indices, self.inner.has_subtree(addr))

    def get_subtree(self, addr):
        raise NotImplementedError

    def get_subtrees_shallow(self):
        raise NotImplementedError

    @typecheck
    def filter(self, tree: ChoiceMap):
        filtered = self.inner.filter(tree)
        flags = jnp.logical_and(
            self.indices >= 0,
            self.indices < static_check_leaf_length(tree),
        )

        def _take(v):
            return jnp.take(v, self.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    def complement(self):
        return ComplementIndexSelection(self)


@dataclass
class ComplementIndexSelection(Selection):
    index_selection: Selection

    def flatten(self):
        return (self.index_selection,), ()

    def get_subtree(self, addr):
        raise NotImplementedError

    def get_subtrees_shallow(self):
        raise NotImplementedError

    def filter(self, tree):
        filtered = self.inner.filter(tree)
        flags = jnp.logical_and(
            self.indices >= 0,
            self.indices < static_check_leaf_length(tree),
        )

        def _take(v):
            return jnp.take(v, self.indices, mode="clip")

        return mask(flags, jtu.tree_map(_take, filtered))

    def complement(self):
        return self.index_selection


##############
# Shorthands #
##############

vector_choice_map = VectorChoiceMap.new
vector_select = VectorSelection.new
index_choice_map = IndexChoiceMap.new
index_select = IndexSelection.new
