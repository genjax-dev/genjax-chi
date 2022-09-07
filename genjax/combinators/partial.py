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
from genjax.builtin.datatypes import JAXGenerativeFunction
from dataclasses import dataclass


@dataclass
class PartialCombinator(GenerativeFunction):
    inner: GenerativeFunction

    def flatten(self):
        return (), (self.inner,)

    def __call__(self, key, *args):
        return self.inner.__call__(key, *args, **self.kwargs)

    @classmethod
    def unflatten(cls, data, xs):
        return PartialCombinator(*data, *xs)

    def simulate(self, key, args):
        end = args[-1]
        closed_over = JAXGenerativeFunction(
            lambda key, *args: self.inner(key, *args, end)
        )
        args = args[0:-1]
        return closed_over.simulate(key, args)

    def importance(self, key, chm, args):
        end = args[-1]
        closed_over = JAXGenerativeFunction(
            lambda key, *args: self.inner(key, *args, end)
        )
        args = args[0:-1]
        return closed_over.importance(key, chm, args)

    def update(self, key, prev, chm, args):
        end = args[-1]
        closed_over = JAXGenerativeFunction(
            lambda key, *args: self.inner(key, *args, end)
        )
        args = args[0:-1]
        return closed_over.update(key, prev, chm, args)
