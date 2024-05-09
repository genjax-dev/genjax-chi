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

from jax.lax import select

from genjax._src.core.generative import (
    Argdiffs,
    Constraint,
    GenerativeFunction,
    Mask,
    MaskedSample,
    MaskedUpdateSpec,
    Retdiff,
    Sample,
    Score,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    BoolArray,
    PRNGKey,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class MaskTrace(Trace):
    mask_combinator: "MaskCombinator"
    inner: Trace
    check: BoolArray

    def get_args(self):
        return (self.check, *self.inner.get_args())

    def get_gen_fn(self):
        return self.mask_combinator

    def get_sample(self):
        return MaskedSample(self.check, self.inner.get_sample())

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
        check, *inner_args = args
        tr = self.gen_fn.simulate(key, tuple(inner_args))
        return MaskTrace(self, tr, check)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[MaskTrace, Weight, UpdateSpec]:
        (check, *inner_args) = args
        tr, w, bwd_spec = self.gen_fn.importance(key, constraint, tuple(inner_args))
        w = check * w
        return (
            MaskTrace(self, tr, check),
            w,
            MaskedUpdateSpec(check, bwd_spec),
        )

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: MaskTrace,
        update_spec: UpdateSpec,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        (check, *_) = Diff.tree_primal(argdiffs)
        (_, *inner_argdiffs) = argdiffs
        inner_trace, w, retdiff, bwd_spec = self.gen_fn.update(
            key, trace.inner, update_spec, tuple(inner_argdiffs)
        )
        w = select(
            check,
            w + trace.get_score(),
            -trace.get_score(),
        )
        return (
            MaskTrace(self, inner_trace, check),
            w,
            Mask.maybe(check, retdiff),
            MaskedUpdateSpec(check, bwd_spec),
        )

    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Mask]:
        (check, *inner_args) = args
        score, retval = self.gen_fn.assess(sample, tuple(inner_args))
        return (
            check * score,
            Mask(check, retval),
        )


#############
# Decorator #
#############


@typecheck
def mask_combinator(
    f: GenerativeFunction,
) -> MaskCombinator:
    return MaskCombinator(f)
