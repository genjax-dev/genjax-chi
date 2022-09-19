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

import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from genjax.core.datatypes import (
    Trace,
    ChoiceMap,
    EmptyChoiceMap,
    ValueChoiceMap,
)
from genjax.core.pytree import squeeze
from genjax.core.specialization import is_concrete
from typing import Union
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp

Bool = Union[jnp.bool_, np.bool_]
Int32 = Union[jnp.int32, np.int32]


@dataclass
class BooleanMask(ChoiceMap):
    mask: Bool
    inner: Union[Trace, ChoiceMap]

    @classmethod
    def new(cls, mask, inner):
        if isinstance(inner, BooleanMask):
            return BooleanMask.new(mask, inner.inner)
        elif isinstance(inner, EmptyChoiceMap):
            return inner
        else:
            return BooleanMask(mask, inner)

    def flatten(self):
        return (self.mask, self.inner), ()

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(
                    indent,
                    pp.concat(
                        [
                            pp.text("mask = "),
                            gpp._pformat(self.mask, **kwargs),
                            pp.brk(),
                            gpp._pformat(self.inner, **kwargs),
                        ]
                    ),
                ),
            ]
        )

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

    def leaf_push(self):
        def _check(v):
            return isinstance(v, ValueChoiceMap) or isinstance(v, BooleanMask)

        return jtu.tree_map(
            lambda v: BooleanMask.new(self.mask, v)
            if isinstance(v, ValueChoiceMap) or isinstance(v, BooleanMask)
            else v,
            self.inner,
            is_leaf=_check,
        )

    def get_retval(self):
        if isinstance(self.inner, Trace):
            return self.inner.get_retval()
        else:
            raise Exception("This BooleanMask does not wrap a Trace.")

    def get_score(self):
        if isinstance(self.inner, Trace):
            return self.inner.get_score()
        else:
            raise Exception("This BooleanMask does not wrap a Trace.")

    def strip_metadata(self):
        return BooleanMask(self.mask, self.inner.strip_metadata())

    @classmethod
    def collapse_boolean_mask(cls, v):
        def _inner(v):
            if isinstance(v, BooleanMask) and is_concrete(v.mask):
                if v.mask:
                    return BooleanMask.collapse_boolean_mask(v.inner)
                else:
                    return EmptyChoiceMap()
            else:
                return v

        def _check(v):
            return isinstance(v, BooleanMask)

        return jtu.tree_map(_inner, v, is_leaf=_check)

    @classmethod
    def boolean_mask_collapse_boundary(cls, fn):
        def _inner(self, key, *args, **kwargs):
            args = BooleanMask.collapse_boolean_mask(args)
            return fn(self, key, *args, **kwargs)

        return _inner

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))


@dataclass
class IndexMask(ChoiceMap):
    index: Int32
    inner: Union[Trace, ChoiceMap]

    def __init__(self, index, inner):
        if isinstance(inner, IndexMask):
            self.inner = inner.inner
        elif isinstance(inner, EmptyChoiceMap):
            return inner
        else:
            self.inner = inner
        self.index = index

    def flatten(self):
        return (self.index, self.inner), ()

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(
                    indent,
                    pp.concat(
                        [
                            pp.text("index: "),
                            gpp._pformat(self.index, **kwargs),
                            pp.brk(),
                            gpp._pformat(self.inner, **kwargs),
                        ]
                    ),
                ),
            ]
        )

    def get_index(self):
        return self.index

    def has_subtree(self, addr):
        return self.inner.has_subtree(addr)

    def get_subtree(self, addr):
        return IndexMask(self.index, squeeze(self.inner.get_subtree(addr)))

    def is_leaf(self):
        return self.inner.is_leaf()

    def get_leaf_value(self):
        return squeeze(self.inner.get_leaf_value())

    def get_subtrees_shallow(self):
        def _inner(k, v):
            return k, IndexMask(self.index, v)

        return map(
            lambda args: _inner(*args),
            self.inner.get_subtrees_shallow(),
        )

    def merge(self, other):
        return squeeze(self.inner.merge(other))

    def leaf_push(self, is_leaf):
        return jtu.tree_map(
            lambda v: IndexMask(self.index, v),
            self.inner,
            is_leaf=lambda v: isinstance(v, ChoiceMap) and v.is_leaf(),
        )

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.index)
        return hash((hash1, hash2))
