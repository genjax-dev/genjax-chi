# Copyright 2023 MIT Probabilistic Computing Project
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
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import mask
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

    @classmethod
    def new(cls, index, submaps):
        return SwitchChoiceMap(index, submaps)

    def is_empty(self):
        # Concrete evaluation -- when possible.
        if all(len(sm) == 0 for sm in self.submaps):
            return True
        else:
            # TODO(colin): hm. I guess this is necessary if the sequence doesn't support random access?
            # Otherwise I'd want len(self.submaps[self.index]) == 0. In fact, why is there an IF here
            # at all?
            flags = jnp.array([len(sm) == 0 for sm in self.submaps])
            return flags[self.index]

    def filter(
        self,
        selection: HierarchicalSelection,
    ) -> ChoiceMap:
        filtered_submaps = map(lambda chm: chm.filter(selection), self.submaps)
        return SwitchChoiceMap(self.index, filtered_submaps)

    def has_submap(self, addr):
        checks = list(map(lambda v: v.has_submap(addr), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    # The way this works is slightly complicated, and relies on specific
    # assumptions about how SwitchCombinator works (and the
    # allowed shapes) of choice maps produced by SwitchCombinator.
    #
    # The key observation is that, if a branch choice map has an addr,
    # and it shares that address with another branch, the shape of the
    # choice map for each shared address has to be the same, all the
    # way down to the arguments.
    def get_submap(self, addr):
        submaps = list(map(lambda v: v.get_submap(addr), self.submaps))

        # Here, we create an index map before we filter out empty ChoiceMaps.
        counter = 0
        index_map = []
        for v in submaps:
            # TODO(colin): here's a point where the behavior could differ betweeen EmptyChoice and an empty HierarchicalChoiceMap.
            if len(v) == 0:
                index_map.append(-1)
            else:
                index_map.append(counter)
                counter += 1
        index_map = jnp.array(index_map)

        non_empty_submaps = list(filter(lambda v: len(v) > 0, submaps))
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

    def get_selection(self):
        raise NotImplementedError

    @dispatch
    def merge(self, other: ChoiceMap) -> Tuple[ChoiceMap, ChoiceMap]:
        new_submaps, new_discard = list(
            zip(*map(lambda v: v.merge(other), self.submaps))
        )
        return SwitchChoiceMap.new(self.index, list(new_submaps)), SwitchChoiceMap.new(
            self.index, list(new_discard)
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        doc = gpp._pformat_array(self.index, short_arrays=True)
        tree = Tree(f"[bold](Switch,{doc})")
        for submap in self.submaps:
            submap_tree = submap.__rich_tree__()
            tree.add(submap_tree)
        return tree


################
# Switch trace #
################


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
        return self.chm.strip()

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
        subtrace = self.chm.submaps[concrete_index]
        return subtrace
