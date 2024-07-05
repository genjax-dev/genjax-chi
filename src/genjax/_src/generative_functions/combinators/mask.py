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
    IncrementalUpdateRequest,
    Mask,
    MaskedSample,
    MaskedUpdateRequest,
    Retdiff,
    Sample,
    Score,
    Trace,
    UpdateRequest,
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
    """
    Combinator which enables dynamic masking of generative functions. Takes a [`genjax.GenerativeFunction`][] and returns a new [`genjax.GenerativeFunction`][] which accepts an additional boolean first argument.

    If `True`, the invocation of the generative function is masked, and its contribution to the score is ignored. If `False`, it has the same semantics as if one was invoking the generative function without masking.

    The return value type is a `Mask`, with a flag value equal to the supplied boolean.

    Parameters:
        gen_fn: The generative function to be masked.

    Returns:
        The masked version of the input generative function.

    Examples:
        Masking a normal draw:
        ```python exec="yes" html="true" source="material-block" session="mask"
        import genjax, jax


        @genjax.mask
        @genjax.gen
        def masked_normal_draw(mean):
            return genjax.normal(mean, 1.0) @ "x"


        key = jax.random.PRNGKey(314159)
        tr = jax.jit(masked_normal_draw.simulate)(
            key,
            (
                False,
                2.0,
            ),
        )
        print(tr.render_html())
        ```
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
        update_request: UpdateRequest,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        (check, *_) = Diff.tree_primal(argdiffs)
        (check_diff, *inner_argdiffs) = argdiffs
        match trace:
            case MaskTrace():
                inner_trace = trace.inner
            case EmptyTrace():
                inner_trace = EmptyTrace(self.gen_fn)

        premasked_trace, w, retdiff, bwd_problem = self.gen_fn.update(
            key,
            inner_trace,
            IncrementalUpdateRequest(tuple(inner_argdiffs), update_request),
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
            MaskedUpdateRequest(check, bwd_problem),
        )

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: UpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        match update_request:
            case IncrementalUpdateRequest(argdiffs, subrequest):
                return self.update_change_target(key, trace, subrequest, argdiffs)
            case _:
                return self.update_change_target(
                    key, trace, update_request, Diff.no_change(trace.get_args())
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
def mask(f: GenerativeFunction) -> GenerativeFunction:
    """
    Combinator which enables dynamic masking of generative functions. Takes a [`genjax.GenerativeFunction`][] and returns a new [`genjax.GenerativeFunction`][] which accepts an additional boolean first argument.

    If `True`, the invocation of the generative function is masked, and its contribution to the score is ignored. If `False`, it has the same semantics as if one was invoking the generative function without masking.

    The return value type is a `Mask`, with a flag value equal to the supplied boolean.

    Args:
        f: The generative function to be masked.

    Returns:
        The masked version of the input generative function.

    Examples:
        Masking a normal draw:
        ```python exec="yes" html="true" source="material-block" session="mask"
        import genjax, jax


        @genjax.mask
        @genjax.gen
        def masked_normal_draw(mean):
            return genjax.normal(mean, 1.0) @ "x"


        key = jax.random.PRNGKey(314159)
        tr = jax.jit(masked_normal_draw.simulate)(
            key,
            (
                False,
                2.0,
            ),
        )
        print(tr.render_html())
        ```
    """
    return MaskCombinator(f)
