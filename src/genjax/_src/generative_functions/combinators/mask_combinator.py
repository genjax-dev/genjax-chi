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
    Constraint,
    GenerativeFunction,
    Mask,
    MaskSample,
    Retdiff,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    BoolArray,
    FloatArray,
    PRNGKey,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class MaskTrace(Trace):
    mask_combinator: "MaskCombinator"
    args: Tuple
    inner: Trace
    check: BoolArray

    def get_args(self):
        return self.args

    def get_gen_fn(self):
        return self.mask_combinator

    def get_sample(self):
        return MaskSample(self.check, self.inner.get_sample())

    def get_retval(self):
        return Mask(self.check, self.inner.get_retval())

    def get_score(self):
        return self.check * self.inner.get_score()


@Pytree.dataclass
class MaskCombinator(GenerativeFunction):
    """A combinator which enables dynamic masking of generative functions.
    `MaskCombinator` takes a `GenerativeFunction` as a parameter, and
    returns a new `GenerativeFunction` which accepts a boolean array as the
    first argument denoting if the invocation of the generative function should
    be masked or not.

    The return value type is a `Mask`, with a flag value equal to the passed in boolean array.

    If the invocation is masked with the boolean array `False`, it's contribution to the score of the trace is ignored. Otherwise, it has same semantics as if one was invoking the generative function without masking.
    """

    gen_fn: GenerativeFunction

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> MaskTrace:
        tr = self.gen_fn.simulate(key, args)
        return MaskTrace(self, tr, self.check)

    @typecheck
    def assess(
        self,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[FloatArray, Mask]:
        score, retval = self.gen_fn.assess(constraint, args)
        return self.check * score, Mask(self.check, retval)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[MaskTrace, FloatArray]:
        tr, w = self.gen_fn.importance(key, constraint, args)
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

            case Constraint():
                inner_gen_fn = self.inner(*self.inner_args)
                inner_trace, w, retdiff, bwd_spec = inner_gen_fn.update(
                    key, trace.inner, update_spec
                )
                return (
                    MaskTrace(self, inner_trace, self.check),
                    w * self.check,
                    Mask.maybe(self.check, retdiff),
                    UpdateSpec.maybe(self.check, bwd_spec),
                )

            case _:
                raise ValueError(f"Unsupported update spec {update_spec}")


#############
# Decorator #
#############


def mask_combinator(f) -> MaskCombinator:
    return MaskCombinator(f)
