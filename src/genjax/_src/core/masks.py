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

import jax.tree_util as jtu

from genjax._src.core.pytree import Pytree
from genjax._src.core.tree import Leaf
from genjax._src.core.typing import Any
from genjax._src.core.typing import Bool


@dataclass
class Mask(Pytree):
    @classmethod
    def new(cls, mask, inner):
        if isinstance(inner, cls):
            return cls.new(mask, inner.unmask())
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

            # `Leaf` inheritors have a method `set_leaf_value`
            # to participate in masking.
            # They can choose how to construct themselves after
            # being provided with a masked value.
            elif isinstance(v, Leaf):
                leaf_value = v.get_leaf_value()
                if isinstance(leaf_value, BooleanMask):
                    return v.set_leaf_value(BooleanMask(self.mask, leaf_value.unmask()))
                else:
                    return v.set_leaf_value(BooleanMask(self.mask, leaf_value))
            else:
                return v

        def _check(v):
            return isinstance(v, BooleanMask) or isinstance(v, Leaf)

        return jtu.tree_map(_inner, self.inner, is_leaf=_check)

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))


##############
# Shorthands #
##############

mask = BooleanMask.new
