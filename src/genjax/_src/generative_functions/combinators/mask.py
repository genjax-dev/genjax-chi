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
    ChoiceMap,
    ChoiceMapConstraint,
    EditRequest,
    GenerativeFunction,
    IncrementalGenericRequest,
    Mask,
    MaskedSample,
    Projection,
    Retdiff,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import Constraint
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

    def get_sample(self) -> MaskedSample:
        return MaskedSample(self.check, self.inner.get_sample())

    def get_choices(self) -> ChoiceMap:
        inner_choice_map = self.inner.get_choices()
        return inner_choice_map.mask(self.check)

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

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[MaskTrace[R], Weight]:
        check, inner_args = args[0], args[1:]
        check = Flag.as_flag(check)

        tr, w = self.gen_fn.generate(key, constraint, inner_args)
        return MaskTrace(self, tr, check), w * check

    def project(
        self,
        key: PRNGKey,
        trace: Trace[R],
        projection: Projection[Any],
    ) -> Weight:
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[Mask[R]],
        edit_request: EditRequest,
    ) -> tuple[MaskTrace[R], Weight, Retdiff[Mask[R]], EditRequest]:
        assert isinstance(trace, MaskTrace)
        assert isinstance(edit_request, IncrementalGenericRequest)
        argdiffs = edit_request.argdiffs
        (check_arg, *rest) = argdiffs
        inner_trace = trace.inner
        subrequest = IncrementalGenericRequest(tuple(rest), edit_request.constraint)
        edited_trace, weight, retdiff, bwd_move = inner_trace.edit(key, subrequest)

        #       What's the math for the weight term here?
        #
        # Well, if we started with a "masked false trace",
        # and then we flip the check_arg to True, we can re-use
        # the sampling process which created the original trace as
        # part of the move. The weight is the entire new trace's score.
        #
        # That's the transition False -> True:
        #
        #               w' = final_trace.score()
        #
        # On the other hand, if we started True, and went False, no matter
        # the update, we can make the choice that this move is just removing
        # the samples from the original trace, and ignoring the move.
        #
        # That's the transition True -> False:
        #
        #               w' = -original_trace.score()
        #
        # For the transition False -> False, we just ignore the move entirely.
        #
        #               w' = 0.0
        #
        # For the transition True -> True, we apply the move to the existing
        # unmasked trace. In that case, the weight is just the weight of the move.
        #
        #               w' = w
        #
        # In any case, we always apply the move... we're not avoiding
        # that computation

        check = trace.check
        final_trace = jtu.tree_map(
            lambda v1, v2: jnp.where(check_arg, v1, v2), edited_trace, inner_trace
        )
        inner_chm = bwd_move.constraint.choice_map
        return (
            MaskTrace(self, final_trace, trace.check),
            (check and check_arg) * weight
            + (check and not check_arg) * (-inner_trace.get_score())
            + (not check and not check_arg) * 0.0
            + (not check and check_arg) * final_trace.get_score(),
            retdiff,
            IncrementalGenericRequest(
                trace.get_args(),
                ChoiceMapConstraint(inner_chm.mask(check_arg)),
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
