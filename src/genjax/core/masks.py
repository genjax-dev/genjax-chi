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
This module contains a set of utility types for "masking" :code:`ChoiceTree`-based
data (like :code:`ChoiceMap` implementors).

This masking functionality is designed to support dynamic control flow concepts
in Gen modeling languages (e.g. :code:`SwitchCombinator`).
"""

from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax.core.choice_tree import ChoiceTree
from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import Trace
from genjax.core.pytree import squeeze
from genjax.core.specialization import is_concrete


Bool = Union[jnp.bool_, np.bool_]
Int32 = Union[jnp.int32, np.int32]


@dataclass
class Mask(ChoiceMap):
    @classmethod
    def new(cls, mask, inner):
        if isinstance(inner, cls):
            return cls.new(mask, inner.inner)
        elif isinstance(inner, EmptyChoiceMap):
            return inner
        else:
            return cls(mask, inner)

    @classmethod
    def collapse_boundary(cls, fn):
        def _inner(self, key, *args, **kwargs):
            args = cls.collapse(args)
            args = tuple(
                map(
                    lambda v: v.leaf_push() if isinstance(v, cls) else v,
                    args,
                )
            )
            return fn(self, key, *args, **kwargs)

        return _inner

    @classmethod
    def canonicalize(cls, fn):
        def __inner(v):
            if isinstance(v, cls):
                return cls.new(jnp.all(v.mask), v.inner)
            else:
                return v

        def _check(v):
            return isinstance(v, cls)

        def _inner(self, key, *args, **kwargs):
            ret = fn(self, key, *args, **kwargs)
            return jtu.tree_map(__inner, ret, is_leaf=_check)

        return _inner


@dataclass
class BooleanMask(Mask):
    mask: Bool
    inner: Union[Trace, ChoiceMap]

    def flatten(self):
        return (self.mask, self.inner), ()

    def has_subtree(self, addr):
        if not self.inner.has_subtree(addr):
            return False
        return self.mask

    def get_subtree(self, addr):
        if not self.inner.has_subtree(addr):
            return EmptyChoiceMap()
        else:
            inner = self.inner.get_subtree(addr)
            return BooleanMask.new(self.mask, inner)

    def is_leaf(self):
        if self.inner.is_leaf():
            return self.mask
        else:
            return False

    def get_leaf_value(self):
        return self.inner.get_leaf_value()

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return k, BooleanMask.new(self.mask, v)

        return map(
            lambda args: _inner(*args), self.inner.get_subtrees_shallow()
        )

    def merge(self, other):
        pushed = self.leaf_push()
        if isinstance(other, BooleanMask):
            return BooleanMask(other.mask, pushed.inner.merge(other.inner))
        return pushed.merge(other)

    def get_retval(self):
        return self.inner.get_retval()

    def get_score(self):
        return self.inner.get_score()

    def get_choices(self):
        return BooleanMask(self.mask, self.inner.get_choices())

    def strip(self):
        return BooleanMask(self.mask, self.inner.strip())

    def leaf_push(self):
        return jtu.tree_map(
            lambda v: BooleanMask.new(self.mask, v),
            self.inner,
            is_leaf=lambda v: isinstance(v, ChoiceMap) and v.is_leaf(),
        )

    @classmethod
    def collapse(cls, v):
        def _inner(v):
            if isinstance(v, BooleanMask) and is_concrete(v.mask):
                if v.mask:
                    return BooleanMask.collapse(v.inner)
                else:
                    return EmptyChoiceMap()
            else:
                return v

        def _check(v):
            return isinstance(v, BooleanMask)

        if isinstance(v, BooleanMask):
            return jtu.tree_map(_inner, v, is_leaf=_check)
        else:
            return v

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))


@dataclass
class IndexMask(Mask):
    index: Int32
    inner: Union[Trace, ChoiceTree]

    def flatten(self):
        return (self.index, self.inner), ()

    def get_index(self):
        return self.index

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return IndexMask.new(self.index, squeeze(self.inner.get_subtree(addr)))

    def is_leaf(self):
        return self.inner.is_leaf()

    def get_leaf_value(self):
        return squeeze(self.inner.get_leaf_value())

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return k, IndexMask.new(self.index, v)

        return map(
            lambda args: _inner(*args),
            self.inner.get_subtrees_shallow(),
        )

    def merge(self, other):
        if isinstance(other, IndexMask):
            return IndexMask.new(
                self.index, squeeze(self.inner.merge(other.inner))
            )
        else:
            return IndexMask.new(self.index, squeeze(self.inner.merge(other)))

    def leaf_push(self):
        return jtu.tree_map(
            lambda v: IndexMask.new(self.index, v),
            self.inner,
            is_leaf=lambda v: isinstance(v, ChoiceMap) and v.is_leaf(),
        )

    # TODO: revisit design.
    def get_selection(self):
        return IndexMask.new(self.index, self.inner.get_selection())

    def complement(self):
        return IndexMask.new(self.index, self.inner.complement())

    def filter(self, chm):
        inner, w = self.inner.filter(chm)
        inner = squeeze(inner)
        inner = jtu.tree_map(lambda v: v[..., self.index], inner)
        return IndexMask.new(self.index, inner), w

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.index)
        return hash((hash1, hash2))
