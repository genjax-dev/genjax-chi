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
from genjax.core.pytree import Pytree
import genjax.core.pretty_printer as pp

#####
# GenerativeFunction
#####


@dataclass
class GenerativeFunction(Pytree, metaclass=abc.ABCMeta):
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

    # Implement the `Pytree` interface methods.
    @abc.abstractmethod
    def flatten(self):
        pass

    @classmethod
    @abc.abstractmethod
    def unflatten(cls, data, xs):
        pass

    def simulate(self, key, args):
        pass

    def importance(self, key, chm, args):
        pass

    def diff(self, key, original, new, args):
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
# ChoiceMap
#####


@dataclass
class ChoiceMap(Pytree, metaclass=abc.ABCMeta):
    def get_choices(self):
        return self

    # Implement the `Pytree` interface methods.
    @abc.abstractmethod
    def flatten(self):
        pass

    @classmethod
    @abc.abstractmethod
    def unflatten(cls, data, xs):
        pass

    @abc.abstractmethod
    def has_choice(self, addr):
        pass

    @abc.abstractmethod
    def get_choice(self, addr):
        pass

    @abc.abstractmethod
    def get_choices_shallow(self):
        pass

    @abc.abstractmethod
    def map(self, fn):
        pass

    def to_selection(self):
        return NoneSelection()

    def __repr__(self):
        return pp.tree_pformat(self)

    def __str__(self):
        return pp.tree_pformat(self)

    def __getitem__(self, k):
        return self.get_choice(k)


@dataclass
class EmptyChoiceMap(ChoiceMap):
    def flatten(self):
        return (), ()

    @classmethod
    def unflatten(cls, data, xs):
        return EmptyChoiceMap()

    def has_choice(self, k):
        return False

    def get_choice(self, k):
        raise Exception("EmptyChoiceMap does not address any values.")

    def get_choices_shallow(self):
        return ()

    def map(self, fn):
        return fn(())


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


#####
# Trace
#####


@dataclass
class Trace(Pytree, metaclass=abc.ABCMeta):
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

    def get_choices_shallow(self):
        choices = self.get_choices()
        return choices.get_choices_shallow()

    def strip_metadata(self):
        return self.get_choices().strip_metadata()

    def __repr__(self):
        return pp.tree_pformat(self)

    def __str__(self):
        return pp.tree_pformat(self)
