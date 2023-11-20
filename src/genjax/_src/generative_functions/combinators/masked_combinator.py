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

from dataclasses import dataclass

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import mask
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import typecheck


@dataclass
class MaskedTrace(Trace):
    mask_combinator: "MaskedCombinator"
    inner: Trace
    check: BoolArray

    def flatten(self):
        return (self.mask_combinator, self.inner, self.check), ()

    def get_gen_fn(self):
        return self.mask_combinator

    def get_choices(self):
        return mask(self.check, self.inner.get_choices())

    def get_retval(self):
        return mask(self.check, self.inner.get_retval())

    def get_score(self):
        return self.check * self.inner.get_score()

    def get_args(self):
        return (self.check, *self.inner.get_args())


@dataclass
class MaskedCombinator(GenerativeFunction):
    inner: GenerativeFunction

    def flatten(self):
        return (self.inner,), ()

    @typecheck
    @classmethod
    def new(cls, gen_fn: GenerativeFunction):
        return MaskedCombinator(gen_fn)

    def simulate(self, key, args):
        (check, inner_args) = args
        tr = self.inner.simulate(key, inner_args)
        return MaskedTrace(self, tr, check)

    def importance(self, key, choice_map, args):
        (check, inner_args) = args
        w, tr = self.inner.importance(key, choice_map, inner_args)
        w = check * w
        return w, MaskedTrace(check, tr)

    def update(self, key, prev_trace, choice_map, argdiffs):
        pass


Masked = MaskedCombinator.new
