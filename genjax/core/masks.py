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

import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from genjax.core.datatypes import Trace, ChoiceMap, EmptyChoiceMap
from genjax.core.pytree import squeeze
from typing import Union
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp

Bool = Union[jnp.bool_, np.bool_]
Int32 = Union[jnp.int32, np.int32]


@dataclass
class BooleanMask(ChoiceMap):
    inner: Union[Trace, ChoiceMap]
    mask: Bool

    def __init__(self, inner, mask):
        if isinstance(inner, BooleanMask):
            self.inner = inner.inner
        elif isinstance(inner, EmptyChoiceMap):
            return inner
        else:
            self.inner = inner
        self.mask = mask

    def flatten(self):
        return (self.inner, self.mask), ()

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(indent, gpp._pformat(self.inner, **kwargs)),
                pp.brk(),
                pp.concat(
                    [
                        pp.text("index = "),
                        gpp._pformat(self.mask, **kwargs),
                    ]
                ),
            ]
        )

    def has_choice(self, addr):
        if not self.inner.has_choice(addr):
            return False
        return self.mask

    def get_choice(self, addr):
        if not self.inner.has_choice(addr):
            return EmptyChoiceMap()
        else:
            inner = self.inner.get_choice(addr)
            return BooleanMask(inner, self.mask)

    def has_value(self):
        if self.inner.has_value():
            return self.mask
        else:
            return False

    def get_value(self):
        assert self.inner.has_value()
        return self.inner.get_value()

    def get_choices_shallow(self):
        def _inner(k, v):
            return k, BooleanMask(v, self.mask)

        return map(lambda args: _inner(*args), self.inner.get_choices_shallow())

    def merge(self, other):
        pushed = self.leaf_push()
        return pushed.merge(other)

    def leaf_push(self):
        return jtu.tree_map(
            lambda v: BooleanMask(v, self.mask),
            self.inner,
            is_leaf=lambda v: isinstance(v, ChoiceMap) and v.has_value(),
        )

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))


@dataclass
class IndexMask(ChoiceMap):
    inner: Union[Trace, ChoiceMap]
    index: Int32

    def __init__(self, inner, index):
        if isinstance(inner, IndexMask):
            self.inner = inner.inner
        elif isinstance(inner, EmptyChoiceMap):
            return inner
        else:
            self.inner = inner
        self.index = index

    def flatten(self):
        return (self.inner, self.index), ()

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(indent, gpp._pformat(self.inner, **kwargs)),
                pp.brk(),
                pp.concat(
                    [
                        pp.text("index = "),
                        gpp._pformat(self.index, **kwargs),
                    ]
                ),
            ]
        )

    def get_index(self):
        return self.index

    def has_choice(self, addr):
        return self.inner.has_choice(addr)

    def get_choice(self, addr):
        return IndexMask(squeeze(self.inner.get_choice(addr)), self.index)

    def has_value(self):
        return self.inner.has_value()

    def get_value(self):
        return squeeze(self.inner.get_value())

    def get_choices_shallow(self):
        def _inner(k, v):
            return k, IndexMask(v, self.index)

        return map(lambda args: _inner(*args), self.inner.get_choices_shallow())

    def merge(self, other):
        return squeeze(self.inner.merge(other))

    def leaf_push(self, is_leaf):
        return jtu.tree_map(
            lambda v: IndexMask(v, self.index),
            self.inner,
            is_leaf=lambda v: isinstance(v, ChoiceMap) and v.has_value(),
        )

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.index)
        return hash((hash1, hash2))
