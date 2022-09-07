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
import collections.abc
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from genjax.core.pytree import Pytree, squeeze
from genjax.core.specialization import concrete_cond, is_concrete
import genjax.core.pretty_printer as pp
from typing import Any, Sequence

#####
# GenerativeFunction
#####


@dataclass
class GenerativeFunction(
    Pytree,
    collections.abc.Callable,
    collections.abc.Hashable,
):
    """
    `GenerativeFunction` class which allows user-defined
    implementations of the generative function interface methods, rather
    than the JAX-driven tracing implementation
    (as provided for the builtin modeling language).

    The implementation will interact with the JAX tracing machinery,
    however, so there are specific API requirements -- enforced via
    Python abstract base class methods.

    The user *must* match the interface signatures of the native JAX
    implementation. This is not statically checked - but failure to do so
    will lead to unintended behavior or errors.

    To support argument and choice gradients via JAX, the user must
    provide a differentiable `importance` implementation.
    """

    def simulate(self, key, args):
        pass

    def importance(self, key, chm, args):
        pass

    def update(self, key, original, new, args):
        pass

    def arg_grad(self, key, tr, args, argnums):
        pass

    def choice_grad(self, key, tr, chm, args):
        pass

    def __repr__(self):
        return pp.tree_pformat(self)

    def __str__(self):
        return pp.tree_pformat(self)


#####
# Trace
#####


@dataclass
class Trace(Pytree, collections.abc.Hashable):
    @abc.abstractmethod
    def get_retval(self):
        pass

    @abc.abstractmethod
    def get_score(self):
        pass

    @abc.abstractmethod
    def get_args(self):
        pass

    @abc.abstractmethod
    def get_choices(self):
        pass

    @abc.abstractmethod
    def get_gen_fn(self):
        pass

    def has_choice(self, addr):
        choices = self.get_choices()
        return choices.has_choice(addr)

    def get_choice(self, addr):
        choices = self.get_choices()
        return choices.get_choice(addr)

    def has_value(self):
        choices = self.get_choices()
        return choices.has_value()

    def get_value(self):
        choices = self.get_choices()
        return choices.get_value()

    def get_choices_shallow(self):
        choices = self.get_choices()
        return choices.get_choices_shallow()

    def strip_metadata(self):
        return self.get_choices().strip_metadata()

    def __repr__(self):
        return pp.tree_pformat(self)

    def __str__(self):
        return pp.tree_pformat(self)

    def __getitem__(self, addr):
        if isinstance(addr, slice):
            return jax.tree_util.tree_map(lambda v: v[addr], self)
        choices = self.get_choices()
        choice = choices.get_choice(addr)
        if choice.has_value():
            return choice.get_value()
        else:
            return choice


#####
# ChoiceMap
#####


@dataclass
class ChoiceMap(Pytree, collections.abc.Hashable):
    @abc.abstractmethod
    def has_choice(self, addr):
        pass

    @abc.abstractmethod
    def get_choice(self, addr):
        pass

    @abc.abstractmethod
    def has_value(self):
        pass

    @abc.abstractmethod
    def get_value(self):
        pass

    @abc.abstractmethod
    def get_choices_shallow(self):
        pass

    def to_selection(self):
        return NoneSelection()

    @abc.abstractmethod
    def merge(self, other):
        pass

    def get_choices(self):
        return self

    def slice(self, arr: Sequence):
        return jtu.tree_map(
            lambda v: v[arr],
            self,
        )

    def __repr__(self):
        return pp.tree_pformat(self)

    def __str__(self):
        return pp.tree_pformat(self)

    def __getitem__(self, addr):
        if isinstance(addr, slice):
            return jax.tree_util.tree_map(lambda v: v[addr], self)
        choice = self.get_choice(addr)
        if choice.has_value():
            return choice.get_value()
        else:
            return choice

    def __eq__(self, other):
        return self.flatten() == other.flatten()


@dataclass
class EmptyChoiceMap(ChoiceMap):
    def flatten(self):
        return (), ()

    @classmethod
    def unflatten(cls, data, xs):
        return EmptyChoiceMap()

    def has_choice(self, addr):
        return False

    def get_choice(self, addr):
        raise Exception("EmptyChoiceMap does not address any values.")

    def has_value(self):
        return False

    def get_value(self):
        raise Exception("EmptyChoiceMap is not a value choice map.")

    def get_choices_shallow(self):
        return ()

    def merge(self, other):
        return other

    def strip_metadata(self):
        return self


@dataclass
class Mask(ChoiceMap, Trace):
    inner: Any
    mask: Any

    def __init__(self, inner, mask):
        self.inner = inner
        self.mask = mask

    def flatten(self):
        return (self.inner, self.mask), ()

    @classmethod
    def unflatten(cls, data, xs):
        return Mask(*data, *xs)

    def has_choice(self, addr):
        if not self.mask.has_choice(addr):
            return False
        check = self.mask.get_choice(addr)
        if isinstance(check, ChoiceMap) and check.has_value():
            value = check.get_value()
            if isinstance(value, np.ndarray):
                return np.any(value)
            elif isinstance(value, jnp.ndarray):
                return jnp.any(value)
            else:
                return value
        else:
            return concrete_cond(
                check,
                lambda *addr: True,
                lambda *addr: False,
            )

    def get_choice(self, addr):
        if not self.inner.has_choice(addr):
            return EmptyChoiceMap()
        else:
            check = self.mask.get_choice(addr)
            inner = self.inner.get_choice(addr)
            return Mask(inner, check)

    def has_value(self):
        mask_chm = self.mask.get_choices()
        if mask_chm.has_value():
            check = mask_chm.get_value()
            if isinstance(check, np.ndarray):
                return np.any(check)
            elif isinstance(check, jnp.ndarray):
                return jnp.any(check)
            return check
        else:
            return False

    def get_value(self):
        return self.inner.get_value()

    def get_choices_shallow(self):
        def _inner(k, v):
            return k, Mask(v, self.mask.get_choice(k))

        return map(lambda args: _inner(*args), self.inner.get_choices_shallow())

    def merge(self, other):
        if isinstance(other, Mask):
            return Mask(
                self.inner.merge(other.inner), self.mask.merge(other.mask)
            )
        else:
            mask_other = mask(other, True)
            return self.merge(mask_other)

    def get_retval(self):
        if isinstance(self.inner, Trace):
            return self.inner.get_retval()
        else:
            raise Exception("Mask does not mask a trace.")

    def get_args(self):
        if isinstance(self.inner, Trace):
            return self.inner.get_args()
        else:
            raise Exception("Mask does not mask a trace.")

    def get_score(self):
        if isinstance(self.inner, Trace):
            return self.inner.get_score()
        else:
            raise Exception("Mask does not mask a trace.")

    def get_gen_fn(self):
        if isinstance(self.inner, Trace):
            return self.inner.get_gen_fn()
        else:
            raise Exception("Mask does not mask a trace.")

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.mask)
        return hash((hash1, hash2))


# This is quite complicated -- it's sort of a "postwalk" style
# pattern which traverses `Pytree`-structured data and simultaneously unwraps
# inner `Mask` instances (pulling `Mask` to the toplevel wrapper) as well
# as filling the mask with `fill`.
def mask(v, fill):
    if isinstance(v, Mask):
        new_mask = jtu.tree_map(lambda v: fill, v.inner)
        return Mask(v.inner, new_mask)
    else:
        sub = jtu.tree_map(
            lambda v: v.inner if isinstance(v, Mask) else v,
            v,
            is_leaf=lambda v: isinstance(v, Mask),
        )
        new_mask = jtu.tree_map(
            lambda v: mask(v, True).mask if isinstance(v, Mask) else fill,
            v,
            is_leaf=lambda v: isinstance(v, Mask),
        )
        return Mask(sub, new_mask)


def collapse_mask(chm):
    def _inner(v):
        if isinstance(v, Mask):
            if is_concrete(v.has_value()):
                return v.inner
            else:
                return v
        else:
            return v

    return jtu.tree_map(_inner, chm, is_leaf=lambda v: isinstance(v, Mask))


@dataclass
class IndexedMask(ChoiceMap):
    inner: Any
    mask: Sequence

    def __init__(self, inner, mask):
        self.inner = inner
        self.mask = mask

    def flatten(self):
        return (self.inner, self.mask), ()

    @classmethod
    def unflatten(cls, data, xs):
        return IndexedMask(*data, *xs)

    def has_choice(self, addr):
        return self.inner.has_choice(addr)

    def get_choice(self, addr):
        return squeeze(self.inner.slice(self.mask)).get_choice(addr)

    def has_value(self):
        return self.inner.slice(self.mask).has_value()

    def get_value(self):
        return squeeze(self.inner.slice(self.mask)).get_value()

    def get_choices_shallow(self):
        return squeeze(self.inner.slice(self.mask)).get_choices_shallow()

    def merge(self, other):
        return squeeze(self.inner.slice(self.mask)).merge(other)

    def __hash__(self):
        hash1 = hash(self.inner)
        hash2 = hash(self.mask.tostring())
        return hash((hash1, hash2))


#####
# Selection
#####


@dataclass
class Selection(Pytree):
    # Implement the `Pytree` interface methods.
    @abc.abstractmethod
    def flatten(self):
        pass

    @classmethod
    @abc.abstractmethod
    def unflatten(cls, data, xs):
        pass

    @abc.abstractmethod
    def filter(self, chm):
        pass

    @abc.abstractmethod
    def complement(self):
        pass

    def __repr__(self):
        return pp.tree_pformat(self)

    def __str__(self):
        return pp.tree_pformat(self)


@dataclass
class AllSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def unflatten(cls, data, xs):
        return AllSelection()

    def filter(self, chm):
        return chm

    def complement(self):
        return NoneSelection()


@dataclass
class NoneSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def unflatten(cls, data, xs):
        return AllSelection()

    def filter(self, chm):
        return EmptyChoiceMap()

    def complement(self):
        return AllSelection()
