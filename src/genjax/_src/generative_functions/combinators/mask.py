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
    ChoiceMap,
    EmptyTrace,
    GenerativeFunction,
    GenericProblem,
    Mask,
    MaskedProblem,
    MaskedSample,
    Retdiff,
    Sample,
    Score,
    Trace,
    UpdateProblem,
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
        inner_sample = self.inner.get_sample()
        if isinstance(inner_sample, ChoiceMap):
            return ChoiceMap.maybe(self.check, inner_sample)
        else:
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
    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        (check, *_) = Diff.tree_primal(argdiffs)
        (check_diff, *inner_argdiffs) = argdiffs
        match trace:
            case MaskTrace():
                inner_trace = trace.inner
            case EmptyTrace():
                inner_trace = EmptyTrace(self.gen_fn)

        premasked_trace, w, retdiff, bwd_problem = self.gen_fn.update(
            key, inner_trace, GenericProblem(tuple(inner_argdiffs), update_problem)
        )
        w = select(
            check,
            w + trace.get_score(),
            -trace.get_score(),
        )
        return (
            MaskTrace(self, premasked_trace, check),
            w,
            Mask.maybe(check_diff, retdiff),
            MaskedProblem(check, bwd_problem),
        )

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case GenericProblem(argdiffs, subproblem):
                return self.update_change_target(key, trace, subproblem, argdiffs)
            case _:
                return self.update_change_target(
                    key, trace, update_problem, Diff.no_change(trace.get_args())
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
def mask(f: GenerativeFunction) -> MaskCombinator:
    return MaskCombinator(f)
