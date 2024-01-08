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
from typing import Union

import jax.numpy as jnp
from jaxtyping import ArrayLike

from genjax._src.core.datatypes.generative import (
    AllSelection,
    Choice,
    ChoiceMap,
    ChoiceValue,
    DisjointUnionChoiceMap,
    EmptyChoice,
    HierarchicalChoiceMap,
    HierarchicalSelection,
    Selection,
)
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pytree.checks import (
    static_check_tree_leaves_have_matching_leading_dim,
)
from genjax._src.core.typing import IntArray
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexedChoiceMap,
    IndexedSelection,
    VectorChoiceMap,
)


def trie_from_dict(constraints: dict):
    trie = Trie()
    for k, v in constraints.items():
        if isinstance(v, dict):
            trie[k] = trie_from_dict(v)
        else:
            trie[k] = choice_map(v)
    return trie

def choice_map(*vs) -> ChoiceMap:
    """Shortcut constructor for GenJAX ChoiceMap objects.

    When called with no arguments, returns an empty (mutable) choice map which
    you can populate using the subscript operator as in

        ```python
        chm = genjax.choice_map()
        chm["x"] = 3.0
        ```

    When called with a dictionary argument, the equivalent HierarchicalChoiceMap
    will be created and returned. (Exception: in the event that all the keys in
    the dictory are integers, an IndexedChoiceMap is produced.)

    When called with a single argument of any other type, constructs a ChoiceValue.

    Finally, if called with a sequence of other ChoiceMap objects, produces a
    DisjointUnionChoiceMap.
    """
    if len(vs) == 0:
        return HierarchicalChoiceMap()
    elif len(vs) == 1:
        v = vs[0]
        if isinstance(v, Choice):
            return v
        elif isinstance(v, dict):
            if all(isinstance(k, int) for k in v.keys()):
                return IndexedChoiceMap.from_dict(v)
            else:
                return HierarchicalChoiceMap(trie_from_dict(v))
        else:
            return ChoiceValue(v)
    else:
        if not all(map(lambda m: isinstance(m, ChoiceMap), vs)):
            raise TypeError(
                "To create a union ChoiceMap, all arguments must be ChoiceMaps"
            )
        return DisjointUnionChoiceMap(*vs)


def indexed_choice_map(ks: ArrayLike, inner: ChoiceMap) -> Union[IndexedChoiceMap, EmptyChoice]:
    if isinstance(inner, EmptyChoice):
        return inner

    indices = jnp.array(ks, copy=False)
    static_check_tree_leaves_have_matching_leading_dim((inner, indices))
    return IndexedChoiceMap(jnp.array(ks), choice_map(inner))

def vector_choice_map(c):
    if isinstance(c, EmptyChoice):
        return c
    elif isinstance(c, ChoiceMap):
        return VectorChoiceMap(c)
    elif isinstance(c, dict):
        return VectorChoiceMap(choice_map(c))
    else:
        raise NotImplementedError(f"Creating VectorChoiceMap from {type(c)}")

def indexed_select(idx: Union[int, IntArray], *choices: Selection):
    idx = jnp.atleast_1d(idx)
    if len(choices) == 0:
        return IndexedSelection(idx, AllSelection())
    elif len(choices) == 1 and isinstance(choices[0], Selection):
        return IndexedSelection(idx, choices[0])
    else:
        return IndexedSelection(idx, HierarchicalSelection.from_addresses(choices))


__all__ = [
    "choice_map",
    "indexed_choice_map",
    "vector_choice_map",
    "indexed_select"
]