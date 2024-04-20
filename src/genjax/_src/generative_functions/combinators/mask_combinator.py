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
    ChoiceMap,
    JAXGenerativeFunction,
    Mask,
    Selection,
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


class MaskTrace(Trace):
    mask_combinator: "MaskCombinator"
    inner: Trace
    check: BoolArray

    def get_gen_fn(self):
        return self.mask_combinator

    def get_choices(self):
        return ChoiceMap.m(self.check, self.inner.get_choices())

    def get_retval(self):
        return Mask(self.check, self.inner.get_retval())

    def get_score(self):
        return self.check * self.inner.get_score()

    def get_args(self):
        return (self.check, self.inner.get_args())

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        return self.check * self.inner.project(key, selection)


class MaskCombinator(JAXGenerativeFunction, SupportsCalleeSugar):
    """A combinator which enables dynamic masking of generative function.
    `MaskCombinator` takes a `GenerativeFunction` as a parameter, and
    returns a new `GenerativeFunction` which accepts a boolean array as the
    first argument denoting if the invocation of the generative function should
    be masked or not.

    The return value type is a `Mask`, with a flag value equal to the passed in boolean array.

    If the invocation is masked with the boolean array `False`, it's contribution to the score of the trace is ignored. Otherwise, it has same semantics as if one was invoking the generative function without masking.
    """

    inner: JAXGenerativeFunction

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> MaskTrace:
        (check, *inner_args) = args
        tr = self.inner.simulate(key, tuple(inner_args))
        return MaskTrace(self, tr, check)

    @typecheck
    def assess(
        self,
        choice: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Mask]:
        (check, *inner_args) = args
        score, retval = self.inner.assess(choice, inner_args)
        return check * score, Mask(check, retval)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        choice: ChoiceMap,
        args: Tuple,
    ) -> Tuple[MaskTrace, FloatArray]:
        (check, *inner_args) = args
        tr, w = self.inner.importance(key, choice, inner_args)
        w = check * w
        return MaskTrace(self, tr, check), w

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev_trace: MaskTrace,
        choice: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[MaskTrace, FloatArray, Any, ChoiceMap]:
        (check_diff, *inner_argdiffs) = argdiffs
        check = Diff.tree_primal(check_diff)
        tr, w, rd, d = self.inner.update(key, prev_trace.inner, choice, inner_argdiffs)
        return (
            MaskTrace(self, tr, check),
            w * check,
            Mask.maybe(check, rd),
            ChoiceMap.m(check, d),
        )

    @property
    def __wrapped__(self):
        return self.inner


#############
# Decorator #
#############


def mask_combinator(f) -> MaskCombinator:
    return module_update_wrapper(MaskCombinator(f))
