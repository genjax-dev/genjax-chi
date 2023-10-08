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

import itertools
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import TraceType
from genjax._src.core.datatypes.masking import mask
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import Sequence
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch


###############################
# Switch combinator datatypes #
###############################

#####
# SwitchChoiceMap
#####

# Note that the abstract/concrete semantics of `jnp.choose`
# are slightly interesting. If we know ahead of time that
# the index is concrete, we can use `jnp.choose` without a
# fallback mode (e.g. index is out of bounds).
#
# If we do not know the index array ahead of time, we must
# choose a fallback mode to allow tracer values.


@dataclass
class SwitchChoiceMap(ChoiceMap):
    index: IntArray
    submaps: Sequence[ChoiceMap]

    def flatten(self):
        return (self.index, self.submaps), ()

    def is_empty(self):
        # Concrete evaluation -- when possible.
        if all(map(lambda sm: isinstance(sm, EmptyChoiceMap), self.submaps)):
            return True
        else:
            flags = jnp.array([sm.is_empty() for sm in self.submaps])
            return flags[self.index]

    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        filtered_submaps = map(lambda chm: chm.filter(selection), self.submaps)
        return SwitchChoiceMap(self.index, filtered_submaps)

    def has_subtree(self, addr):
        checks = list(map(lambda v: v.has_subtree(addr), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    # The way this works is slightly complicated, and relies on specific
    # assumptions about how SwitchCombinator works (and the
    # allowed shapes) of choice maps produced by SwitchCombinator.
    #
    # The key observation is that, if a branch choice map has an addr,
    # and it shares that address with another branch, the shape of the
    # choice map for each shared address has to be the same, all the
    # way down to the arguments.
    def get_subtree(self, addr):
        submaps = list(map(lambda v: v.get_subtree(addr), self.submaps))

        # Here, we create an index map before we filter out
        # EmptyChoiceMap instances.
        counter = 0
        index_map = []
        for v in submaps:
            if isinstance(v, EmptyChoiceMap):
                index_map.append(-1)
            else:
                index_map.append(counter)
                counter += 1
        index_map = jnp.array(index_map)

        non_empty_submaps = list(
            filter(lambda v: not isinstance(v, EmptyChoiceMap), submaps)
        )
        indexer = index_map[self.index]

        def chooser(*trees):
            shapediff = len(trees[0].shape) - len(indexer.shape)
            reshaped = indexer.reshape(indexer.shape + (1,) * shapediff)
            return jnp.choose(reshaped, trees, mode="wrap")

        # TODO: A bit of broadcast wizardry, would be good
        # to make this understandable.
        flags = (jnp.arange(len(non_empty_submaps))[:, None] == indexer.flatten()).sum(
            axis=-1, dtype=bool
        )

        return mask(
            flags[indexer],
            jtu.tree_map(
                chooser,
                *non_empty_submaps,
            ),
        )

    def get_subtrees_shallow(self):
        def _inner(index, submap):
            check = index == self.index
            return map(
                lambda v: (v[0], mask(check, v[1])),
                submap.get_subtrees_shallow(),
            )

        sub_iterators = map(
            lambda args: _inner(*args),
            enumerate(self.submaps),
        )
        return itertools.chain(*sub_iterators)

    def get_selection(self):
        raise NotImplementedError

    @dispatch
    def merge(self, other: ChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        new_submaps, new_discard = list(
            zip(*map(lambda v: v.merge(other), self.submaps))
        )
        return SwitchChoiceMap.new(self.index, new_submaps), SwitchChoiceMap.new(
            self.index, new_discard
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.index, short_arrays=True)
        sub_tree = Tree(f"[bold](Switch,{doc})")
        for submap in self.submaps:
            sub_root = Tree("")
            submap_tree = submap.__rich_tree__(sub_root)
            sub_tree.add(submap_tree)
        tree.add(sub_tree)
        return tree


#####
# SwitchTrace
#####


@dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    chm: SwitchChoiceMap
    args: Tuple
    retval: Any
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.chm,
            self.args,
            self.retval,
            self.score,
        ), ()

    def get_args(self):
        return self.args

    def get_choices(self):
        return self.chm

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def project(self, selection: Selection) -> FloatArray:
        weights = list(map(lambda v: v.project(selection), self.chm.submaps))
        return jnp.choose(self.chm.index, weights, mode="wrap")

    def get_subtrace(self, concrete_index):
        switch_chm = self.get_choices()
        subtrace = switch_chm.submaps[concrete_index]
        return subtrace


#####
# SumTraceType
#####


@dataclass
class SumTraceType(TraceType):
    summands: Sequence[TraceType]

    def flatten(self):
        return (), (self.summands,)

    def is_leaf(self):
        return all(map(lambda v: v.is_leaf(), self.summands))

    def get_leaf_value(self):
        pass

    def has_subtree(self, addr):
        return any(map(lambda v: v.has_subtree(addr), self.summands))

    def get_subtree(self, addr):
        pass

    def get_subtrees_shallow(self):
        sub_iterators = map(
            lambda v: v.get_subtrees_shallow(),
            self.summands,
        )
        return itertools.chain(*sub_iterators)

    def merge(self, other):
        raise Exception("Not implemented.")

    def __subseteq__(self, other):
        return False

    def get_rettype(self):
        return self.summands[0].get_rettype()
