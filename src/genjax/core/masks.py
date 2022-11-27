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

import abc

from dataclasses import dataclass
from typing import Union, Any

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.datatypes import Trace
from genjax.core.pytree import squeeze, Pytree

Bool = Union[jnp.bool_, np.bool_]


@dataclass
class Mask(Pytree):
    @classmethod
    def new(cls, mask, inner):
        if isinstance(inner, cls):
            return cls.new(mask, inner.unmask())
        elif isinstance(inner, EmptyChoiceMap):
            return inner
        else:
            return cls(mask, inner).leaf_push()

    @abc.abstractmethod
    def leaf_push(self):
        pass

    @abc.abstractmethod
    def unmask(self):
        pass


@dataclass
class BooleanMask(Mask):
    mask: Bool
    inner: Any

    def flatten(self):
        return (self.mask, self.inner), ()

    def unmask(self):
        return self.inner

    def leaf_push(self):
        def _inner(v):
            if isinstance(v, BooleanMask):
                return BooleanMask.new(self.mask, v.unmask())
            elif isinstance(v, ChoiceMap) and v.is_leaf():
                leaf_value = v.get_leaf_value()
                if isinstance(leaf_value, BooleanMask):
                    return v.new(BooleanMask(self.mask, leaf_value.unmask()))
                else:
                    return v.new(BooleanMask(self.mask, leaf_value))
            else:
                return v

        def _check(v):
            return isinstance(v, BooleanMask) or (
                isinstance(v, ChoiceMap) and v.is_leaf()
            )

        return jtu.tree_map(_inner, self.inner, is_leaf=_check)

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))
