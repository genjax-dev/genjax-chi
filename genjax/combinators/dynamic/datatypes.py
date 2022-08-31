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
This module implements a generative function combinator which allows
choice map dynamic lookups. The generative function implementations
utilize :code:`trace` handlers which are implemented with `jax.lax.cond`.

The main point of usage for this functionality is  the efficient
implemention of both the `Unfold` and `Map` combinators.
"""

from genjax.builtin import JAXTrace
from genjax.core.datatypes import (
    GenerativeFunction,
)
from .handlers import (
    simulate,
    importance,
)
from dataclasses import dataclass
from typing import Callable

#####
# DynamicJAXGenerativeFunction
#####


@dataclass
class DynamicJAXGenerativeFunction(GenerativeFunction):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    @classmethod
    def unflatten(cls, data, xs):
        return DynamicJAXGenerativeFunction(*data, *xs)

    def simulate(self, key, args, **kwargs):
        key, (f, args, r, chm, score) = simulate(self.source)(
            key, args, **kwargs
        )
        return key, JAXTrace(f, args, r, chm, score)

    def importance(self, key, mask, chm, args, **kwargs):
        key, (w, (f, args, r, chm, score)) = importance(self.source)(
            key, mask, chm, args, **kwargs
        )
        return key, (w, JAXTrace(f, args, r, chm, score))
