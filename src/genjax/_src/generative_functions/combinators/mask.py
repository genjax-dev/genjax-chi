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
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    ChoiceMapConstraint,
    Constraint,
    EditRequest,
    GenerativeFunction,
    IncrementalChoiceMapRequest,
    Mask,
    Projection,
    Retdiff,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import FlagOp
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Generic,
    PRNGKey,
    ScalarFlag,
    TypeVar,
)

R = TypeVar("R")


@Pytree.dataclass
class MaskTrace(Generic[R], Trace[Mask[R]]):
    mask_combinator: "MaskCombinator[R]"
    inner: Trace[R]
    check: ScalarFlag

    def get_args(self) -> tuple[Any, ...]:
        return (self.check, *self.inner.get_args())

    def get_gen_fn(self):
        return self.mask_combinator

    def get_sample(self) -> ChoiceMap:
        return self.get_choices()

    def get_choices(self) -> ChoiceMap:
        inner_choice_map = self.inner.get_choices()
        return inner_choice_map.mask(self.check)

    def get_retval(self):
        return Mask(self.inner.get_retval(), self.check)

    def get_score(self):
        inner_score = self.inner.get_score()
        return jnp.asarray(
            FlagOp.where(self.check, inner_score, jnp.zeros(shape=inner_score.shape))
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

        tr = self.gen_fn.simulate(key, inner_args)
        return MaskTrace(self, tr, check)

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[MaskTrace[R], Weight]:
        check, inner_args = args[0], args[1:]

        tr, w = self.gen_fn.generate(key, constraint, inner_args)
        return MaskTrace(self, tr, check), w * check

    def project(
        self,
        key: PRNGKey,
        trace: Trace[Mask[R]],
        projection: Projection[Any],
    ) -> Weight:
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[Mask[R]],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[MaskTrace[R], Weight, Retdiff[Mask[R]], EditRequest]:
        assert isinstance(trace, MaskTrace)
        assert isinstance(edit_request, IncrementalChoiceMapRequest)

        check_diff, inner_argdiffs = argdiffs[0], argdiffs[1:]
        post_check: ScalarFlag = Diff.tree_primal(check_diff)

        match trace:
            case MaskTrace():
                pre_check = trace.check
                original_trace: Trace[R] = trace.inner

        subrequest = IncrementalChoiceMapRequest(edit_request.constraint)

        premasked_trace, weight, retdiff, bwd_request = self.gen_fn.edit(
            key, original_trace, subrequest, inner_argdiffs
        )

        final_trace: Trace[R] = jtu.tree_map(
            lambda v1, v2: jnp.where(post_check, v1, v2),
            premasked_trace,
            original_trace,
        )

        t_to_t = FlagOp.and_(pre_check, post_check)
        t_to_f = FlagOp.and_(pre_check, FlagOp.not_(post_check))
        f_to_f = FlagOp.and_(FlagOp.not_(pre_check), FlagOp.not_(post_check))
        f_to_t = FlagOp.and_(FlagOp.not_(pre_check), post_check)

        final_weight = (
            #       What's the math for the weight term here?
            #
            # Well, if we started with a "masked false trace",
            # and then we flip the check_arg to True, we can re-use
            # the sampling process which created the original trace as
            # part of the move. The weight is the entire new trace's score.
            #
            # That's the transition False -> True:
            #
            #               final_weight = final_trace.score()
            #
            f_to_t * final_trace.get_score()
            #
            # On the other hand, if we started True, and went False, no matter
            # the update, we can make the choice that this move is just removing
            # the samples from the original trace, and ignoring the move.
            #
            # That's the transition True -> False:
            #
            #               final_weight = -original_trace.score()
            #
            + t_to_f * -original_trace.get_score()
            #
            # For the transition False -> False, we just ignore the move entirely.
            #
            #               final_weight = 0.0
            #
            + f_to_f * 0.0
            #
            # For the transition True -> True, we apply the move to the existing
            # unmasked trace. In that case, the weight is just the weight of the move.
            #
            #               final_weight = weight
            #
            + t_to_t * weight
            #
            # In any case, we always apply the move... we're not avoiding
            # that computation.
        )

        assert isinstance(bwd_request, IncrementalChoiceMapRequest)
        inner_chm_constraint = bwd_request.constraint
        assert isinstance(inner_chm_constraint, ChoiceMapConstraint)

        return (
            MaskTrace(self, premasked_trace, post_check),
            final_weight,
            Mask.maybe(retdiff, check_diff),
            IncrementalChoiceMapRequest(
                ChoiceMapConstraint(inner_chm_constraint.mask(post_check)),
            ),
        )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, Mask[R]]:
        (check, *inner_args) = args
        score, retval = self.gen_fn.assess(sample, tuple(inner_args))
        return (
            check * score,
            Mask(retval, check),
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
