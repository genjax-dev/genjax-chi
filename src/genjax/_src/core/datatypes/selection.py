# Copyright 2024 MIT Probabilistic Computing Project
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


import re
from abc import abstractmethod

import jax.numpy as jnp
import rich.tree as rich_tree

from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any, BoolArray, IntArray, TraceSliceComponent, Tuple


class Selection(Pytree):
    @abstractmethod
    def complement(self) -> "Selection":
        """Return a `Selection` which filters addresses to the complement set of the
        provided `Selection`.

        Examples:
            ```python exec="yes" source="tabbed-left"
            import jax
            import genjax
            from genjax import bernoulli

            console = genjax.console()


            @genjax.static_gen_fn
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x


            key = jax.random.PRNGKey(314159)
            tr = model.simulate(key, ())
            choice = tr.strip()
            selection = genjax.select("x")
            complement = selection.complement()
            filtered = choice.filter(complement)
            print(console.render(filtered))
            ```
        """
        pass


class MapSelection(Selection):
    def complement(self) -> "MapSelection":
        return ComplementMapSelection(self)

    @abstractmethod
    def get_subselection(self, addr) -> "Selection":
        raise NotImplementedError

    @abstractmethod
    def has_addr(self, addr) -> BoolArray:
        raise NotImplementedError

    ###########
    # Dunders #
    ###########

    def __getitem__(self, addr):
        subselection = self.get_subselection(addr)
        return subselection


class ComplementMapSelection(MapSelection):
    selection: Selection

    def complement(self):
        return self.selection

    def has_addr(self, addr):
        assert isinstance(self.selection, MapSelection)
        return not self.selection.has_addr(addr)

    def get_subselection(self, addr):
        assert isinstance(self.selection, MapSelection)
        return self.selection.get_subselection(addr).complement()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](Complement)")
        tree.add(self.selection.__rich_tree__())
        return tree


#######################
# Concrete selections #
#######################


class NoneSelection(Selection):
    def complement(self):
        return AllSelection()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](NoneSelection)")
        return tree


class AllSelection(Selection):
    def complement(self):
        return NoneSelection()

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        return rich_tree.Tree("[bold](AllSelection)")


class TraceSlice(Selection):
    s: Tuple[TraceSliceComponent, ...]

    def __init__(self, s: Tuple[TraceSliceComponent, ...] | TraceSliceComponent):
        if not isinstance(s, tuple):
            s = (s,)
        self.s = s

    def has_addr(self, item: Any):
        if len(self.s) == 0:
            # If the selection tuple is exhausted, it matches everything "below"
            return True
        s0 = self.s[0]
        if isinstance(s0, str):
            if isinstance(item, str):
                return re.fullmatch(s0, item)
        elif isinstance(s0, set):
            return item in set
        raise NotImplementedError(f"match slice element {s0!r} with {item!r}")

    def get_subselection(self, k: str) -> "TraceSlice":
        """This is an admittedly unprincipled method, since in the slice model,
        we aren't holding a map of subselections. Generally, when this method
        is used, it follows a use of `has_addr(k)`, and is therefore safe, but
        TODO we should find a way to eliminate the parameter."""
        return self.inner

    @property
    def indices(self) -> IntArray | slice:
        """Return a JNP array of the integer indices contained within this selection."""
        if not self.s:
            return jnp.array([])
        s0 = self.s[0]
        if isinstance(s0, jnp.ndarray) or isinstance(s0, slice):
            return s0
        if isinstance(s0, int):
            return jnp.array([s0])

        raise NotImplementedError(
            f"address component {s0} cannot be converted to index array"
        )

    @property
    def inner(self) -> "TraceSlice":
        return TraceSlice(self.s[1:])

    def complement(self) -> Selection:
        raise NotImplementedError("cannot invert slice selection")


###########################
# Concrete map selections #
###########################


class HierarchicalSelection(MapSelection):
    trie: Trie

    @classmethod
    def from_addresses(cls, *addresses: Any):
        trie = Trie()
        for addr in addresses:
            trie = trie.trie_insert(addr, AllSelection())
        return HierarchicalSelection(trie)

    def has_addr(self, addr):
        return self.trie.has_submap(addr)

    def get_subselection(self, addr):
        value = self.trie.get_submap(addr)
        if value is None:
            return NoneSelection()
        else:
            subselect = value
            if isinstance(subselect, Trie):
                return HierarchicalSelection(subselect)
            else:
                return subselect

    # Extra method which is useful to generate an iterator
    # over keys and subselections at the first level.
    def get_subselections_shallow(self):
        def _inner(v):
            addr = v[0]
            submap = v[1].get_selection()
            if isinstance(submap, Trie):
                submap = HierarchicalSelection(submap)
            return (addr, submap)

        return map(
            _inner,
            self.trie.get_submaps_shallow(),
        )

    ###################
    # Pretty printing #
    ###################

    def __rich_tree__(self):
        tree = rich_tree.Tree("[bold](HierarchicalSelection)")
        for k, v in self.get_subselections_shallow():
            subk = tree.add(f"[bold]:{k}")
            subk.add(v.__rich_tree__())
        return tree
