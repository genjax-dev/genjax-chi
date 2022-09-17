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

import jax.numpy as jnp
import numpy as np
import itertools
from genjax.core.masks import BooleanMask
from genjax.core.datatypes import ChoiceMap, Trace, EmptyChoiceMap
from dataclasses import dataclass
from typing import Union, Sequence
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp

Int = Union[jnp.int32, np.int32]

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


#####
# IndexedChoiceMap
#####

# Note that the abstract/concrete semantics of `jnp.choose`
# are slightly interesting. If we know ahead of time that
# the index is concrete, we can use `jnp.choose` without a
# fallback mode (e.g. index is out of bounds).
#
# If we do not know the index array ahead of time, we must
# choose a fallback mode to allow tracer values.


@dataclass
class IndexedChoiceMap(ChoiceMap):
    index: Int
    submaps: Sequence[Union[ChoiceMap, Trace]]

    def flatten(self):
        return (self.index, self.submaps), ()

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(
                    indent,
                    pp.concat(
                        [
                            pp.text("index = "),
                            gpp._pformat(self.index, **kwargs),
                            pp.brk(),
                            gpp._pformat(self.submaps, **kwargs),
                        ]
                    ),
                ),
            ]
        )

    def is_leaf(self):
        checks = list(map(lambda v: v.is_leaf(), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    def get_leaf_value(self):
        leafs = list(
            map(
                lambda v: jnp.array(False)
                if (not v.is_leaf()) or isinstance(v, EmptyChoiceMap)
                else v.get_leaf_value(),
                self.submaps,
            )
        )
        return jnp.choose(self.index, leafs, mode="wrap")

    def has_subtree(self, addr):
        checks = list(map(lambda v: v.has_subtree(addr), self.submaps))
        return jnp.choose(self.index, checks, mode="wrap")

    def get_subtree(self, addr):
        submaps = list(map(lambda v: v.get_subtree(addr), self.submaps))
        return IndexedChoiceMap(self.index, submaps)

    def get_subtrees_shallow(self):
        def _inner(index, submap):
            check = index == self.index
            return map(
                lambda v: (v[0], BooleanMask.new(check, v[1])),
                submap.get_subtrees_shallow(),
            )

        sub_iterators = map(
            lambda args: _inner(*args),
            enumerate(self.submaps),
        )
        return itertools.chain(*sub_iterators)

    def merge(self, other):
        new_submaps = list(map(lambda v: v.merge(other), self.submaps))
        return IndexedChoiceMap(self.index, new_submaps)
