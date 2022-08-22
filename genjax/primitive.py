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
Exposes a `PrimitiveGenerativeFunction` class which can be used to
encapsulate domain-specific implementations of the generative function
interface methods.

Supporting this functionality is slightly complicated -- it requires
introducing a new primitive `extern_p` (see `intrinsics.py`) -- when
a user defines a new `PrimitiveGenerativeFunction`, they are allowed to
provide method implementations for the generative function interface.

These method implementation must support JAX tracing
(because we're calling the `PrimitiveGenerativeFunction` inside a modeling
language which utilizes JAX tracing to handle code generation).

Thus, a user must provide an `abstract_eval` method -- to support JAX's abstract evaluation.
"""

import abc
from genjax.core import Pytree


class PrimitiveGenerativeFunction(Pytree, metaclass=abc.ABCMeta):
    """
    `PrimitiveGenerativeFunction` class which allows user-defined
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

    # Interact with JAX's abstract tracer.
    @classmethod
    @abc.abstractmethod
    def abstract_eval(cls, key, *args, **kwargs):
        pass

    # Implement any subset of GFI methods.
    def simulate(self, key, args):
        pass

    def importance(self, key, chm, args):
        pass

    def diff(self, key, original, new, args):
        pass

    def update(self, key, original, new, args):
        pass
