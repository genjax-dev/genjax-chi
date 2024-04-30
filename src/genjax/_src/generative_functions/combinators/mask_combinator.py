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

from genjax._src.core.generative import (
    ChangeTargetUpdateSpec,
    ChoiceMap,
    Constraint,
    GenerativeFunction,
    Mask,
    Retdiff,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    BoolArray,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
    typecheck,
)


@Pytree.dataclass
class MaskTrace(Trace):
    mask_combinator: "MaskCombinator"
    inner: Trace
    check: BoolArray

    def get_gen_fn(self):
        return self.mask_combinator

    def get_sample(self):
        return ChoiceMap.m(self.check, self.inner.get_sample())

    def get_retval(self):
        return Mask(self.check, self.inner.get_retval())

    def get_score(self):
        return self.check * self.inner.get_score()


@Pytree.dataclass
class MaskCombinator(GenerativeFunction):
    """A combinator which enables dynamic masking of generative function.
    `MaskCombinator` takes a `GenerativeFunction` as a parameter, and
    returns a new `GenerativeFunction` which accepts a boolean array as the
    first argument denoting if the invocation of the generative function should
    be masked or not.

    The return value type is a `Mask`, with a flag value equal to the passed in boolean array.

    If the invocation is masked with the boolean array `False`, it's contribution to the score of the trace is ignored. Otherwise, it has same semantics as if one was invoking the generative function without masking.
    """

    check: BoolArray
    inner_args: Tuple
    inner: Callable[[Any], GenerativeFunction] = Pytree.static()

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
    ) -> MaskTrace:
        inner_gen_fn = self.inner(*self.inner_args)
        tr = inner_gen_fn.simulate(key)
        return MaskTrace(self, tr, self.check)

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
        constraint: Constraint,
    ) -> Tuple[MaskTrace, FloatArray]:
        inner_gen_fn = self.inner(*self.inner_args)
        tr, w = inner_gen_fn.importance(key, constraint)
        w = self.check * w
        return MaskTrace(self, tr, self.check), w

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: MaskTrace,
        update_spec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case ChangeTargetUpdateSpec(argdiffs, constraint):
                inner_gen_fn = self.inner(*self.inner_args)
                (check, _) = Diff.tree_primal(argdiffs)
                inner_spec = ChangeTargetUpdateSpec(argdiffs[1:], constraint)
                inner_trace, w, retdiff, bwd_spec = inner_gen_fn.update(
                    key, trace.inner, inner_spec
                )
                return (
                    MaskTrace(self, inner_trace, check),
                    w * check,
                    Mask.maybe(check, retdiff),
                    UpdateSpec.maybe(check, bwd_spec),
                )


#############
# Decorator #
#############


def mask_combinator(f) -> Callable[[Any], MaskCombinator]:
    def inner(check, *args) -> MaskCombinator:
        return MaskCombinator(check, args, f)

    return inner
