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
    Arguments,
    ChoiceMap,
    ChoiceMapConstraint,
    ChoiceMapEditRequest,
    ChoiceMapProjection,
    ChoiceMapSample,
    EditRequest,
    GenerativeFunction,
    Masked,
    Projection,
    Retdiff,
    Retval,
    Sample,
    Score,
    SelectionProjection,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import Constraint
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    BoolArray,
    Generic,
    PRNGKey,
    TypeVar,
    overload,
)

G = TypeVar("G", bound=GenerativeFunction)
A = TypeVar("A", bound=Arguments)
R = TypeVar("R", bound=Retval)
C = TypeVar("C", bound=Constraint)
S = TypeVar("S", bound=Sample)
P = TypeVar("P", bound=Projection)
Tr = TypeVar("Tr", bound=Trace)
U = TypeVar("U", bound=EditRequest)


@Pytree.dataclass
class MaskedTrace(
    Generic[G, A, R, U],
    Trace[
        "MaskedCombinator[A, R, U]",
        tuple[bool | BoolArray, A],
        ChoiceMapSample,
        Masked[R],
    ],
):
    mask_combinator: "MaskedCombinator[A, R, U]"
    inner: Trace[G, A, ChoiceMapSample, R]
    check: bool | BoolArray

    def get_args(self) -> tuple:
        return (self.check, *self.inner.get_args())

    def get_gen_fn(self) -> "MaskedCombinator[A, R, U]":
        return self.mask_combinator

    def get_sample(self) -> ChoiceMapSample:
        inner_sample = self.inner.get_sample()
        return ChoiceMapSample(ChoiceMap.maybe(self.check, inner_sample))

    def get_retval(self) -> Masked[R]:
        return Masked(self.check, self.inner.get_retval())

    def get_score(self) -> Score:
        return self.check * self.inner.get_score()


@Pytree.dataclass
class MaskedCombinator(
    Generic[A, R, U],
    GenerativeFunction[
        tuple[bool | BoolArray, A],
        ChoiceMapSample,
        Masked[R],
        ChoiceMapConstraint,
        SelectionProjection | ChoiceMapProjection,
        ChoiceMapEditRequest,
    ],
):
    """Combinator which enables dynamic masking of generative functions. Takes
    a [`genjax.GenerativeFunction`][] and returns a new
    [`genjax.GenerativeFunction`][] which accepts an additional boolean first
    argument to indicate whether an invocation of the provided generative
    function should be masked (turning off the generative effects) or not.

    If `True`, the invocation of the generative function is masked, and its contribution to generative computations is ignored. If `False`, it has the same semantics as if one was invoking the generative function without masking.

    The return value type is a `Masked`, with a flag value equal to the supplied boolean.

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

    gen_fn: GenerativeFunction[
        A,
        ChoiceMapSample,
        R,
        ChoiceMapConstraint,
        SelectionProjection | ChoiceMapProjection,
        U,
    ]

    def simulate(
        self,
        key: PRNGKey,
        args: A,
    ) -> MaskedTrace[
        GenerativeFunction[
            A,
            ChoiceMapSample,
            R,
            ChoiceMapConstraint,
            SelectionProjection | ChoiceMapProjection,
            U,
        ],
        A,
        R,
        U,
    ]:
        check, *inner_args = args
        tr = self.gen_fn.simulate(key, tuple(inner_args))
        return MaskedTrace(self, tr, check)

    def assess(
        self,
        key: PRNGKey,
        sample: ChoiceMapSample,
        args: A,
    ) -> tuple[Score, Masked[R]]:
        (check, *inner_args) = args
        score, retval = self.gen_fn.assess(key, sample, tuple(inner_args))
        return (
            score * check,
            Masked.maybe(check, retval),
        )

    def importance_edit(
        self,
        key: PRNGKey,
        constraint: ChoiceMapConstraint,
        args: A,
    ) -> tuple[MaskedTrace, Weight, Projection]:
        (check, *inner_args) = args
        inner_trace, weight, bwd_projection = self.gen_fn.importance_edit(
            key, constraint, tuple(inner_args)
        )
        return (
            MaskedTrace(self, inner_trace, check),
            weight * check,
            ChoiceMapProjection(ChoiceMap.maybe(check, bwd_projection)),
        )

    def project_edit(
        self,
        key: PRNGKey,
        trace: MaskedTrace,
        projection: ChoiceMapProjection | SelectionProjection,
    ) -> tuple[Weight, ChoiceMapConstraint]:
        inner_trace = trace.inner
        weight, fwd_constraint = self.gen_fn.project_edit(key, inner_trace, projection)
        check = trace.check
        return (
            weight * check,
            ChoiceMapConstraint(ChoiceMap.maybe(check, fwd_constraint)),
        )

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: MaskedTrace,
        request: ChoiceMapEditRequest,
        args: A,
    ) -> tuple[MaskedTrace, Weight, Retdiff, ChoiceMapEditRequest]:
        pass

    def edit(
        self,
        key: PRNGKey,
        trace: MaskedTrace,
        request: ChoiceMapEditRequest,
        args: A,
    ) -> tuple[MaskedTrace, Weight, Retdiff, ChoiceMapEditRequest]:
        (check_arg, *rest) = args
        inner_trace = trace.inner
        edited_trace, weight, retdiff, bwd_move = request.edit(
            key, inner_trace, tuple(rest)
        )

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
            MaskedTrace(self, final_trace, trace.check),
            (check and check_arg) * weight
            + (check and not check_arg) * (-inner_trace.get_score())
            + (not check and not check_arg) * 0.0
            + (not check and check_arg) * final_trace.get_score(),
            retdiff,
            ChoiceMapEditRequest(
                ChoiceMapConstraint(ChoiceMap.maybe(check_arg, inner_chm))
            ),
        )


#############
# Decorator #
#############


def mask(f: GenerativeFunction) -> MaskedCombinator:
    """Combinator which enables dynamic masking of generative functions. Takes
    a [`genjax.GenerativeFunction`][] and returns a new
    [`genjax.GenerativeFunction`][] which accepts an additional boolean first
    argument.

    If `True`, the invocation of the generative function is masked, and its contribution to the score is ignored. If `False`, it has the same semantics as if one was invoking the generative function without masking.

    The return value type is a `Masked`, with a flag value equal to the supplied boolean.

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
    return MaskedCombinator(f)
