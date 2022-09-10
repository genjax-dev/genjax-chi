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
import jax
import jax.tree_util as jtu
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp
from dataclasses import dataclass
from genjax.core.pytree import Pytree, squeeze
from genjax.core.tracetypes import Bottom
from typing import Any, Sequence

#####
# GenerativeFunction
#####


@dataclass
class GenerativeFunction(Pytree):
    """
    :code:`GenerativeFunction` abstract class which allows user-defined
    implementations of the generative function interface methods.
    The :code:`builtin` and :code:`distributions` languages both
    implement a class inheritor of :code:`GenerativeFunction`.

    Any implementation will interact with the JAX tracing machinery,
    however, so there are specific API requirements above the requirements
    enforced in other languages (like Gen in Julia). In particular,
    any implementation must provide a :code:`__call__` method so that
    JAX can correctly determine output shapes.

    The user *must* match the interface signatures of the native JAX
    implementation. This is not statically checked - but failure to do so
    will lead to unintended behavior or errors.

    To support argument and choice gradients via JAX, the user must
    provide a differentiable `importance` implementation.
    """

    @abc.abstractmethod
    def __call__(self, key, *args):
        pass

    def get_trace_type(self, key, *args):
        return Bottom()

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
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)


#####
# Trace
#####


@dataclass
class Trace(Pytree):
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

    def overload_pprint(self, **kwargs):
        indent = kwargs["indent"]
        return pp.concat(
            [
                pp.text(f"{type(self)}"),
                pp.brk(""),
                gpp._pformat(self.get_gen_fn(), **kwargs),
                gpp._pformat(self.get_choices(), **kwargs),
                pp.brk(""),
                pp.text(f"score: {self.score}"),
            ]
        )

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
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)

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
class ChoiceMap(Pytree):
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

    @abc.abstractmethod
    def merge(self, other):
        pass

    def to_selection(self):
        return NoneSelection()

    def get_choices(self):
        return self

    def get_score(self):
        return 0.0

    def slice(self, arr: Sequence):
        return squeeze(
            jtu.tree_map(
                lambda v: v[arr],
                self,
            )
        )

    def __repr__(self):
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)

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


#####
# Masking
#####


@dataclass
class BooleanMask(ChoiceMap):
    inner: Any
    mask: Any

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

    @classmethod
    def unflatten(cls, data, xs):
        return BooleanMask(*data, *xs)

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
    inner: Any
    index: Any

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

    @classmethod
    def unflatten(cls, data, xs):
        return IndexMask(*data, *xs)

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
        return squeeze(self.inner.get_choices_shallow())

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
        return gpp.tree_pformat(self)

    def __str__(self):
        return gpp.tree_pformat(self)


@dataclass
class NoneSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def unflatten(cls, data, xs):
        return AllSelection()

    def filter(self, chm):
        return EmptyChoiceMap(), 0.0

    def complement(self):
        return AllSelection()


@dataclass
class AllSelection(Selection):
    def flatten(self):
        return (), ()

    @classmethod
    def unflatten(cls, data, xs):
        return AllSelection()

    def filter(self, chm):
        return chm, chm.get_score()

    def complement(self):
        return NoneSelection()
