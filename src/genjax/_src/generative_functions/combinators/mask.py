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


import jax

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    EmptyTrace,
    GenerativeFunction,
    ImportanceRequest,
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
from genjax._src.core.generative.core import Constraint
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    BoolArray,
    PRNGKey,
)


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

    def simulate(
        self,
        key: PRNGKey,
        args: tuple,
    ) -> MaskTrace:
        check, *inner_args = args
        tr = self.gen_fn.simulate(key, tuple(inner_args))
        return MaskTrace(self, tr, check)

    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: UpdateRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace, Weight, Retdiff, UpdateRequest]:
        (check, *_) = Diff.tree_primal(argdiffs)
        (check_diff, *inner_argdiffs) = argdiffs
        match trace:
            case MaskTrace():
                inner_trace = trace.inner
            case EmptyTrace():
                inner_trace = EmptyTrace(self.gen_fn)
            case _:
                raise NotImplementedError(f"Unexpected trace type: {trace}")

        premasked_trace, w, retdiff, bwd_problem = self.gen_fn.update(
            key,
            inner_trace,
            IncrementalUpdateRequest(tuple(inner_argdiffs), update_request),
        )

        w = jax.lax.select(
            check,
            w,
            -trace.get_score(),
        )

        return (
            MaskTrace(self, premasked_trace, check),
            w,
            Mask.maybe(check_diff, retdiff),
            MaskedUpdateRequest(check, bwd_problem),
        )

    def update_change_target_from_false(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: UpdateRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace, Weight, Retdiff, UpdateRequest]:
        check = Diff.tree_primal(argdiffs)[0]
        check_diff, inner_argdiffs = argdiffs[0], argdiffs[1:]

        inner_trace = EmptyTrace(self.gen_fn)

        assert isinstance(update_request, Constraint)
        imp_update_request = ImportanceRequest(update_request)

        premasked_trace, w, _, _ = self.gen_fn.update(
            key,
            inner_trace,
            IncrementalUpdateRequest(tuple(inner_argdiffs), imp_update_request),
        )

        _, _, retdiff, bwd_problem = self.gen_fn.update(
            key,
            premasked_trace,
            IncrementalUpdateRequest(tuple(inner_argdiffs), update_request),
        )

        w = jax.lax.select(
            check,
            premasked_trace.get_score(),
            0.0,
        )

        return (
            MaskTrace(self, premasked_trace, check),
            w,
            Mask.maybe(check_diff, retdiff),
            MaskedUpdateRequest(check, bwd_problem),
        )

    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_request: UpdateRequest,
    ) -> tuple[Trace, Weight, Retdiff, UpdateRequest]:
        assert isinstance(trace, MaskTrace) or isinstance(trace, EmptyTrace)

        match update_request:
            case IncrementalUpdateRequest(argdiffs, subrequest) if isinstance(
                subrequest, ImportanceRequest
            ):
                return self.update_change_target(key, trace, subrequest, argdiffs)
            case IncrementalUpdateRequest(argdiffs, subrequest):
                assert isinstance(trace, MaskTrace)

                if not trace.check:
                    raise Exception(
                        "This move is not currently supported! See https://github.com/probcomp/genjax/issues/1230 for notes."
                    )

                return jax.lax.cond(
                    trace.check,
                    self.update_change_target,
                    self.update_change_target_from_false,
                    key,
                    trace,
                    subrequest,
                    argdiffs,
                )
            case _:
                return self.update_change_target(
                    key, trace, update_request, Diff.no_change(trace.get_args())
                )

    def assess(
        self,
        sample: Sample,
        args: tuple,
    ) -> tuple[Score, Mask]:
        (check, *inner_args) = args
        score, retval = self.gen_fn.assess(sample, tuple(inner_args))
        return (
            check * score,
            Mask(check, retval),
        )


#############
# Decorator #
#############


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
