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

"""
This module holds a common :code:`ChoiceMap` implementation for
:code:`genjax.MapCombinator` and :code:`genjax.UnfoldCombinator`.

This module also holds a static utility function, which is used in the
generative function implementations for the above combinators to
coerce hierarchical choice maps into forms which can be :code:`scan`d
and :code:`vmap`d.
"""

import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from genjax.core.datatypes import ChoiceMap, Trace
from genjax.core.pytree import tree_stack
from dataclasses import dataclass
from typing import Union
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp

#####
# VectorChoiceMap
#####


@dataclass
class VectorChoiceMap(ChoiceMap):
    inner: Union[ChoiceMap, Trace]

    def flatten(self):
        return (self.inner,), ()

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(indent, gpp._pformat(self.inner, **kwargs)),
            ]
        )

    def is_leaf(self):
        return self.inner.is_leaf()

    def get_leaf_value(self):
        return self.inner.get_leaf_value()

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return self.inner.get_subtree(addr)

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return k, VectorChoiceMap(v)

        return map(
            lambda args: _inner(*args), self.inner.get_subtrees_shallow()
        )

    def merge(self, other):
        return self.inner.merge(other)

    def get_score(self):
        return jnp.sum(self.inner.get_score())

    def __hash__(self):
        return hash(self.inner)


def prepare_vectorized_choice_map(shape, treedef, length, chm):
    chm_vectored = []
    mask_vectored = []
    for k in range(0, length):
        emptied = jtu.tree_map(lambda v: np.zeros((), v.dtype), shape)
        if chm.has_subtree(k):
            submap = chm.get_subtree(k)
            emptied = emptied.merge(submap)
            mask = [
                True if submap.has_subtree(k) else False
                for (k, _) in shape.get_subtrees_shallow()
            ]
            mask = jtu.tree_unflatten(treedef, mask)
        else:
            mask = [False for (k, _) in shape.get_subtrees_shallow()]
            mask = jtu.tree_unflatten(treedef, mask)
        chm_vectored.append(emptied)
        mask_vectored.append(mask)
    mask_vectored = tree_stack(mask_vectored)
    chm_vectored = tree_stack(chm_vectored)
    return chm_vectored, mask_vectored
