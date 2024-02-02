# Copyright 2024 MIT Probabilistic Computing Project
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

from equinox import module_update_wrapper

from genjax._src.core.datatypes.generative import (
    Choice,
    JAXGenerativeFunction,
    Mask,
    Trace,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.typing import (
    Any,
    BoolArray,
    FloatArray,
    PRNGKey,
    Tuple,
    typecheck,
)
from genjax._src.generative_functions.static.static_gen_fn import SupportsCalleeSugar


class MaskingTrace(Trace):
    mask_combinator: "MaskingCombinator"
    inner: Trace
    check: BoolArray

    def get_gen_fn(self):
        return self.mask_combinator

    def get_choice(self):
        return Mask(self.check, self.inner.get_choice())

    def get_retval(self):
        return Mask(self.check, self.inner.get_retval())

    def get_score(self):
        return self.check * self.inner.get_score()

    def get_args(self):
        return (self.check, *self.inner.get_args())


class MaskingCombinator(JAXGenerativeFunction, SupportsCalleeSugar):
    """A combinator which enables dynamic masking of generative function.
    `MaskingCombinator` takes a `GenerativeFunction` as a parameter, and returns a new
    `GenerativeFunction` which accepts a boolean array as the first argument denoting if
    the invocation of the generative function should be masked or not.

    The return value type is a `Mask`, with a flag value equal to the passed in boolean array.

    If the invocation is masked with the boolean array `False`, it's contribution to the score of the trace is ignored. Otherwise, it has same semantics as if one was invoking the generative function without masking.
    """

    inner: JAXGenerativeFunction

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
        check = Diff.tree_primal(check_diff)
        tr, w, rd, d = self.inner.update(key, prev_trace.inner, choice, inner_argdiffs)
        return (
            MaskingTrace(check, tr),
            w * check,
            Mask(check, rd),
            Mask(check, d),
        )

    @property
    def __wrapped__(self):
        return self.inner


#############
# Decorator #
#############


def masking_combinator(f) -> MaskingCombinator:
    return module_update_wrapper(MaskingCombinator(f))
