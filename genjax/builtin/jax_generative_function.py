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

from genjax.core.datatypes import GenerativeFunction
from dataclasses import dataclass
from typing import Callable
from genjax.builtin.jax_choice_map import JAXChoiceMap
from genjax.builtin.jax_trace import JAXTrace
from genjax.builtin.handlers import (
    sample,
    simulate,
    importance,
    diff,
    update,
    arg_grad,
    choice_grad,
)


@dataclass
class JAXGenerativeFunction(GenerativeFunction):
    source: Callable

    def __call__(self, key, *args):
        return self.source(key, *args)

    def flatten(self):
        return (), (self.source,)

    @classmethod
    def unflatten(cls, data, xs):
        return JAXGenerativeFunction(*xs)

    def sample(self, key, args, **kwargs):
        return sample(self.source)(key, args, **kwargs)

    def simulate(self, key, args, **kwargs):
        return simulate(self.source)(key, args, **kwargs)

    def importance(self, key, chm, args, **kwargs):
        return importance(self.source)(key, chm, args, **kwargs)

    def diff(self, key, original, new, args, **kwargs):
        return diff(self.source)(key, original, new, args, **kwargs)

    def update(self, key, original, new, args, **kwargs):
        return update(self.source)(key, original, new, args, **kwargs)

    def arg_grad(self, argnums):
        return lambda key, tr, args, **kwargs: arg_grad(self.source, argnums)(
            key, tr, args, **kwargs
        )

    def choice_grad(self, key, tr, chm, args):
        return choice_grad(self.source)(key, tr, chm, args)


def gen(fn):
    return JAXGenerativeFunction(fn)


ChoiceMap = JAXChoiceMap
Trace = JAXTrace
