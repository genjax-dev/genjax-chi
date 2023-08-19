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
from rich.tree import Tree

import genjax._src.core.pretty_printing as gpp
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
        return cls(tag, values)

    def _static_assert_tagged_union_switch_num_callables_is_num_values(self, callables):
        assert len(callables) == len(self.values)

    def _static_assert_tagged_union_switch_returns_same_type(self, vs):
        return True

    @typecheck
    def switch(self, *callables: Callable):
        assert len(callables) == len(self.values)
        self._static_assert_tagged_union_switch_num_callables_is_num_values(callables)
        vs = list(map(lambda v: v[0](v[1]), zip(callables, self.values)))
        self._static_assert_tagged_union_switch_returns_same_type(vs)
        vs = jnp.array(vs)
        return vs[self.tag]

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self, tree):
        doc = gpp._pformat_array(self.tag, short_arrays=True)
        vals_tree = gpp.tree_pformat(self.values, short_arrays=True)
        sub_tree = Tree(f"[bold](Tagged, {doc})")
        sub_tree.add(vals_tree)
        tree.add(sub_tree)
        return tree

    # Defines custom pretty printing.
    def __rich_console__(self, console, options):
        tree = Tree("")
        tree = self.__rich_tree__(tree)
        yield tree


##############
# Shorthands #
##############

tagged_union = TaggedUnion.new
