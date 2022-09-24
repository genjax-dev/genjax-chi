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
from genjax.builtin.builtin_gen_fn import BuiltinGenerativeFunction
from dataclasses import dataclass
from typing import List


@dataclass
class PartialCombinator(GenerativeFunction):
    inner: GenerativeFunction
    static_argnums: List

    def flatten(self):
        return (), (self.inner, self.static_argnums)

    def __call__(self, key, *args, **kwargs):
        return self.inner.__call__(key, *args, **kwargs)

    def simulate(self, key, args):
        static_args = tuple(
            map(
                lambda ind: args[ind] if ind in self.static_argnums else 0,
                range(0, len(args)),
            )
        )

        def _inner(key, *args):
            new_args = tuple(
                map(
                    lambda ind: static_args[ind]
                    if ind in self.static_argnums
                    else args[ind],
                    range(0, len(args)),
                )
            )
            return self.inner(key, *new_args)

        closed_over = BuiltinGenerativeFunction(_inner)
        return closed_over.simulate(key, args)

    def importance(self, key, chm, args):
        static_args = tuple(
            map(
                lambda ind: args[ind] if ind in self.static_argnums else 0,
                range(0, len(args)),
            )
        )

        def _inner(key, *args):
            new_args = tuple(
                map(
                    lambda ind: static_args[ind]
                    if ind in self.static_argnums
                    else args[ind],
                    range(0, len(args)),
                )
            )
            return self.inner(key, *new_args)

        closed_over = BuiltinGenerativeFunction(_inner)

        return closed_over.importance(key, chm, args)

    def update(self, key, prev, chm, args):
        static_args = tuple(
            map(
                lambda ind: args[ind] if ind in self.static_argnums else 0,
                range(0, len(args)),
            )
        )

        def _inner(key, *args):
            new_args = tuple(
                map(
                    lambda ind: static_args[ind]
                    if ind in self.static_argnums
                    else args[ind],
                    range(0, len(args)),
                )
            )
            return self.inner(key, *new_args)

        closed_over = BuiltinGenerativeFunction(_inner)
        return closed_over.update(key, prev, chm, args)
