# Copyright 2023 MIT Probabilistic Computing Project
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

from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import LanguageConstructor
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import mask
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.typing import Any
from genjax._src.core.typing import BoolArray
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.static.static_gen_fn import SupportsCalleeSugar


@dataclass
class MaskingTrace(Trace):
    mask_combinator: "MaskingCombinator"
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
class MaskingCombinator(JAXGenerativeFunction, SupportsCalleeSugar):
    """A combinator which enables dynamic masking of generative function.
    `MaskingCombinator` takes a `GenerativeFunction` as a parameter, and
    returns a new `GenerativeFunction` which accepts a boolean array as the
    first argument denoting if the invocation of the generative function should
    be masked or not.

    The return value type is a `Mask`, with a flag value equal to the passed in boolean array.

    If the invocation is masked with the boolean array `False`, it's contribution to the score of the trace is ignored. Otherwise, it has same semantics as if one was invoking the generative function without masking.
    """

    inner: JAXGenerativeFunction

    def flatten(self):
        return (self.inner,), ()

    @typecheck
    @classmethod
    def new(cls, gen_fn: JAXGenerativeFunction):
        return MaskingCombinator(gen_fn)

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> MaskingTrace:
        (check, inner_args) = args
        tr = self.inner.simulate(key, inner_args)
        return MaskingTrace(self, tr, check)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        choice: Choice,
        args: Tuple,
    ) -> Tuple[MaskingTrace, FloatArray]:
        (check, inner_args) = args
        w, tr = self.inner.importance(key, choice, inner_args)
        w = check * w
        return MaskingTrace(check, tr), w

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev_trace: MaskingTrace,
        choice: Choice,
        argdiffs: Tuple,
    ) -> Tuple[MaskingTrace, FloatArray, Any, Choice]:
        (check_diff, inner_argdiffs) = argdiffs
        check = tree_diff_primal(check_diff)
        tr, w, rd, d = self.inner.update(key, prev_trace.inner, choice, inner_argdiffs)
        return (
            MaskingTrace(check, tr),
            w * check,
            mask(check, rd),
            mask(check, d),
        )


#########################
# Language constructors #
#########################


@typecheck
def masking_combinator(gen_fn: JAXGenerativeFunction):
    return MaskingCombinator.new(gen_fn)


Masking = LanguageConstructor(
    masking_combinator,
)
