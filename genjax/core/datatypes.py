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
    @classmethod
    @abc.abstractmethod
    def flatten(cls):
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
