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


import jax.numpy as jnp

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    EditRequest,
    EmptyTrace,
    GenerativeFunction,
    ImportanceRequest,
    IncrementalGenericRequest,
    Mask,
    MaskedRequest,
    MaskedSample,
    Retdiff,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import Constraint
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import Flag
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Generic,
    PRNGKey,
    TypeVar,
)

R = TypeVar("R")


@Pytree.dataclass
class MaskTrace(Generic[R], Trace[Mask[R]]):
    mask_combinator: "MaskCombinator[R]"
    inner: Trace[R]
    check: Flag

    def get_args(self) -> tuple[Flag, Any]:
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
        inner_score = self.inner.get_score()
        return jnp.asarray(
            self.check.where(inner_score, jnp.zeros(shape=inner_score.shape))
        )


@Pytree.dataclass
class MaskCombinator(Generic[R], GenerativeFunction[Mask[R]]):
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

    gen_fn: GenerativeFunction[R]

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> MaskTrace[R]:
        check, inner_args = args[0], args[1:]
        check = Flag.as_flag(check)

        tr = self.gen_fn.simulate(key, inner_args)
        return MaskTrace(self, tr, check)

    def edit_change_target(
        self,
        key: PRNGKey,
        trace: Trace[Mask[R]],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[MaskTrace[R], Weight, Retdiff[Mask[R]], EditRequest]:
        check = Diff.tree_primal(argdiffs)[0]
        check_diff, inner_argdiffs = argdiffs[0], argdiffs[1:]
        match trace:
            case MaskTrace():
                inner_trace = trace.inner
            case EmptyTrace():
                inner_trace = EmptyTrace(self.gen_fn)
            case _:
                raise NotImplementedError(f"Unexpected trace type: {trace}")

        premasked_trace, w, retdiff, bwd_problem = self.gen_fn.edit(
            key, inner_trace, IncrementalGenericRequest(inner_argdiffs, edit_request)
        )

        w = check.where(w, -trace.get_score())

        return (
            MaskTrace(self, premasked_trace, check),
            w,
            Mask.maybe(check_diff, retdiff),
            MaskedRequest(check, bwd_problem),
        )

    def edit_change_target_from_false(
        self,
        key: PRNGKey,
        trace: Trace[Mask[R]],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[MaskTrace[R], Weight, Retdiff[Mask[R]], EditRequest]:
        check = Diff.tree_primal(argdiffs)[0]
        check_diff, inner_argdiffs = argdiffs[0], argdiffs[1:]

        inner_trace = EmptyTrace(self.gen_fn)

        assert isinstance(edit_request, Constraint)
        imp_update_problem = ImportanceRequest(edit_request)

        premasked_trace, w, _, _ = self.gen_fn.edit(
            key,
            inner_trace,
            IncrementalGenericRequest(inner_argdiffs, imp_update_problem),
        )

        _, _, retdiff, bwd_problem = self.gen_fn.edit(
            key,
            premasked_trace,
            IncrementalGenericRequest(inner_argdiffs, edit_request),
        )

        premasked_score = premasked_trace.get_score()
        w = check.where(premasked_score, jnp.zeros(premasked_score.shape))

        return (
            MaskTrace(self, premasked_trace, check),
            w,
            Mask.maybe(check_diff, retdiff),
            MaskedRequest(check, bwd_problem),
        )

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[Mask[R]],
        edit_request: EditRequest,
    ) -> tuple[MaskTrace[R], Weight, Retdiff[Mask[R]], EditRequest]:
        assert isinstance(trace, MaskTrace) or isinstance(trace, EmptyTrace)

        match edit_request:
            case IncrementalGenericRequest(argdiffs, subproblem) if isinstance(
                subproblem, ImportanceRequest
            ):
                return self.edit_change_target(key, trace, subproblem, argdiffs)
            case IncrementalGenericRequest(argdiffs, subproblem):
                assert isinstance(trace, MaskTrace)

                if trace.check.concrete_false():
                    raise Exception(
                        "This move is not currently supported! See https://github.com/probcomp/genjax/issues/1230 for notes."
                    )

                return trace.check.cond(
                    self.edit_change_target,
                    self.edit_change_target_from_false,
                    key,
                    trace,
                    subproblem,
                    argdiffs,
                )

            case _:
                return self.edit_change_target(
                    key, trace, edit_request, Diff.no_change(trace.get_args())
                )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, Mask[R]]:
        (check, *inner_args) = args
        score, retval = self.gen_fn.assess(sample, tuple(inner_args))
        return (
            check.f * score,
            Mask(check, retval),
        )


#############
# Decorator #
#############


def mask(f: GenerativeFunction[R]) -> MaskCombinator[R]:
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
