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
    Constraint,
    EditRequest,
    GenerativeFunction,
    Mask,
    Retdiff,
    Score,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.generative.choice_map import Selection
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import FlagOp
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Flag,
    Generic,
    PRNGKey,
    ScalarFlag,
    TypeVar,
)

R = TypeVar("R")


@Pytree.dataclass
class MaskTrace(Generic[R], Trace[Mask[R]]):
    """A trace type for the `MaskCombinator` generative function.

    Users should use `MaskTrace.build` for constructing instances of `MaskTrace`,
    """

    mask_combinator: "MaskCombinator[R]"
    inner: Trace[R]
    args: tuple[Any, ...]
    chm: ChoiceMap
    score: Score
    ret: Mask[R]
    check: Flag

    @staticmethod
    def build(
        scan_gen_fn: "MaskCombinator[R]", inner: Trace[R], check: ScalarFlag
    ) -> "MaskTrace[R]":
        """Construct a new `MaskTrace` instance.

        This static method builds a `MaskTrace` by combining an inner trace with a check flag.
        The check flag determines whether the inner trace's choices and return value are masked.

        Args:
            scan_gen_fn: The MaskCombinator generative function
            inner: The inner trace to be masked
            check: A scalar boolean flag indicating whether to mask the inner trace

        Returns:
            A new MaskTrace instance with choices, return value and score masked with the check flag
        """
        # NOTE: constructing these values in `build`, where `check` is guaranteed scalar, allows us
        # to construct simple, non-vectorized `MaskTrace` instances. Returning these from `jax.vmap`
        # will allow JAX to construct a vectorized `MaskTrace` for us.
        #
        # If we instead deferred these computations to the methods (get_choices() etc), these methods would have to combine vectorized `self.inner.get_choices()` with a vectorized `self.check`. This is tricky and error-prone.
        args = (check, *inner.get_args())
        chm = inner.get_choices().mask(check)
        ret = Mask.build(inner.get_retval(), check)
        score = check * inner.get_score()

        return MaskTrace(scan_gen_fn, inner, args, chm, score, ret, check)

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_gen_fn(self):
        return self.mask_combinator

    def get_choices(self) -> ChoiceMap:
        return self.chm

    def get_retval(self):
        return self.ret

    def get_score(self):
        return self.score


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


        key = jax.random.key(314159)
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
        return MaskTrace.build(self, tr, check)

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[MaskTrace[R], Weight]:
        check, inner_args = args[0], args[1:]

        tr, w = self.gen_fn.generate(key, constraint, inner_args)
        return MaskTrace.build(self, tr, check), w * check

    def project(
        self,
        key: PRNGKey,
        trace: Trace[Mask[R]],
        selection: Selection,
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
        assert isinstance(edit_request, Update)

        check_diff, inner_argdiffs = argdiffs[0], argdiffs[1:]
        post_check: ScalarFlag = Diff.tree_primal(check_diff)

        match trace:
            case MaskTrace():
                pre_check = trace.check
                original_trace: Trace[R] = trace.inner

        subrequest = Update(edit_request.constraint)

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

        assert isinstance(bwd_request, Update)
        inner_chm = bwd_request.constraint

        return (
            MaskTrace.build(self, premasked_trace, post_check),
            final_weight,
            Mask.build(retdiff, check_diff),
            Update(
                inner_chm.mask(post_check),
            ),
        )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, Mask[R]]:
        check, inner_args = args[0], args[1:]
        score, retval = self.gen_fn.assess(sample, inner_args)
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


        key = jax.random.key(314159)
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
