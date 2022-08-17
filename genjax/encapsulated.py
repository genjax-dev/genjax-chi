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
Exposes a `EncapsulatedGenerativeFunction` class which can be used to
encapsulate domain-specific implementations of the generative function
interface methods.

Supporting this functionality is slightly complicated -- it requires
introducing a new primitive `extern_p` (see `intrinsics.py`) -- when
a user defines a new `EncapsulatedGenerativeFunction`, they are allowed to
provide method implementations for the generative function interface.

These method implementation must support JAX tracing
(because we're calling the `EncapsulatedGenerativeFunction` inside a modeling
language which utilizes JAX tracing to handle code generation).

Thus, a user must provide an `abstract_eval` method -- to support JAX's abstract evaluation.
"""

import abc


class EncapsulatedGenerativeFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def simulate(key, args):
        pass

    @abc.abstractmethod
    def importance(key, chm, args):
        pass

    @abc.abstractmethod
    def diff(key, original, new, args):
        pass

    @abc.abstractmethod
    def update(key, original, new, args):
        pass

    @abc.abstractmethod
    def arg_grad(key, tr, args):
        pass

    @abc.abstractmethod
    def choice_grad(key, tr, chm, args):
        pass
