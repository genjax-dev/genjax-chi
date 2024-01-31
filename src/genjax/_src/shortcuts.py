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

import jax.numpy as jnp

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
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import ArrayLike, IntArray, Union, typecheck
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexedChoiceMap,
    IndexedSelection,
    VectorChoiceMap,
)


def trie_from_dict(constraints: dict):
    """Recurses over `constraints`, a Python dictionary, to produce the Trie
    with the same structure.

    Non-dict values are mapped through [[choice_map]].
    """
    trie = Trie()
    for k, v in constraints.items():
        if isinstance(v, dict):
            trie = trie.trie_insert(k, trie_from_dict(v))
        else:
            trie = trie.trie_insert(k, choice_map(v))
    return trie


ChoiceMappable = Union[Choice, dict]


def choice_map(*vs: ChoiceMappable) -> ChoiceMap:
    """Shortcut constructor for GenJAX ChoiceMap objects.

    When called with no arguments, returns an empty (mutable) choice map which
    you can populate using the subscript operator as in

        ```python
        chm = genjax.choice_map()
        chm["x"] = 3.0
        ```

    When called with a dictionary argument, the equivalent :py:class:`HierarchicalChoiceMap`
    will be created and returned. (Exception: in the event that all the keys in
    the dict are integers, an :py:class:`IndexedChoiceMap` is produced.)

    When called with a single argument of any other type, constructs a :py:class:`ChoiceValue`.

    Finally, if called with a sequence of other :py:class:`ChoiceMap` objects, produces a
    :py:class:`DisjointUnionChoiceMap`.
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
        if all(map(lambda m: isinstance(m, ChoiceMap), vs)):
            return DisjointUnionChoiceMap(vs)
        raise TypeError("To create a union ChoiceMap, all arguments must be ChoiceMaps")


def indexed_choice_map(
    ks: ArrayLike, inner: ChoiceMappable
) -> Union[IndexedChoiceMap, EmptyChoice]:
    """Construct an indexed choice map from an array of indices and an inner
    choice map.

    The indices may be a bare integer, or a list or :py:class:`jnp.Array` of integers;
    it will be promoted to a :py:class:`jnp.Array` if needed.

    The inner choice map can of any form accepted by the shortcut py:func:`choice_map`.
    """
    if isinstance(inner, EmptyChoice):
        return inner

    indices = jnp.array(ks, copy=False)
    Pytree.static_check_tree_leaves_have_matching_leading_dim((inner, indices))
    return IndexedChoiceMap(jnp.array(ks), choice_map(inner))


def vector_choice_map(c: ChoiceMappable) -> VectorChoiceMap:
    """Construct a vector choice map from the given one.

    If `c` is the :py:class:`EmptyChoice`, it is returned unmodified; otherwise
    `c` may be of any type accepted by the :py:func:`choice_map` shortcut;
    the result is `VectorChoiceMap(choice_map(c))`.
    """
    if isinstance(c, EmptyChoice):
        return c
    return VectorChoiceMap(choice_map(c))


@typecheck
def indexed_select(idx: Union[int, IntArray], *choices: Selection):
    idx = jnp.atleast_1d(idx)
    if len(choices) == 0:
        return IndexedSelection(idx, AllSelection())
    elif len(choices) == 1 and isinstance(choices[0], Selection):
        return IndexedSelection(idx, choices[0])
    else:
        return IndexedSelection(idx, HierarchicalSelection.from_addresses(choices))
