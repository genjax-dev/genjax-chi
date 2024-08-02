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
    Arguments,
    ChoiceMap,
    EditRequest,
    GenerativeFunction,
    Masked,
    MaskedConstraint,
    MaskedEditRequest,
    MaskedSample,
    Projection,
    Retdiff,
    Retval,
    Sample,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import Constraint, SampleCoercableToChoiceMap
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Bool,
    BoolArray,
    Generic,
    PRNGKey,
    TypeVar,
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
class MaskedTrace(Trace):
    mask_combinator: "MaskedCombinator"
    inner: Trace
    check: Bool | BoolArray

    def get_args(self):
        return (self.check, *self.inner.get_args())

    def get_gen_fn(self) -> "MaskedCombinator":
        return self.mask_combinator

    def get_sample(self) -> MaskedSample:
        inner_sample = self.inner.get_sample()
        return MaskedSample(self.check, self.inner.get_sample())

    def get_choices(self) -> ChoiceMap:
        assert isinstance(self.inner, SampleCoercableToChoiceMap), type(self.inner)
        inner_chm = self.inner.get_choices()
        return ChoiceMap.maybe(self.check, inner_chm)

    def get_retval(self) -> Masked:
        return Masked(self.check, self.inner.get_retval())

    def get_score(self) -> Score:
        return self.check * self.inner.get_score()


@Pytree.dataclass
class MaskedCombinator(
    Generic[Tr, A, S, R, C, P, U],
    GenerativeFunction[
        MaskedTrace,
        tuple[Bool | BoolArray, A],
        S | MaskedSample[S],
        Masked[R],
        C | MaskedConstraint[C, S],
        P,
        U | MaskedEditRequest[U],
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

    gen_fn: GenerativeFunction[Tr, A, S, R, C, P, U]

    def simulate(
        self,
        key: PRNGKey,
        arguments: A,
    ) -> MaskedTrace:
        check, *inner_args = arguments
        tr = self.gen_fn.simulate(key, tuple(inner_args))
        return MaskedTrace(self, tr, check)

    def assess(
        self,
        key: PRNGKey,
        sample: S | MaskedSample[S],
        arguments: A,
    ) -> CouldPanic[tuple[Score, Masked[R]]]:
        (check, *inner_args) = arguments
        match sample:
            case MaskedSample(flag, inner_sample):
                score, retval = self.gen_fn.assess(key, inner_sample, tuple(inner_args))
                check_empty = jnp.logical_and(
                    jnp.logical_not(flag), jnp.logical_not(check)
                )
                check_nonempty = jnp.logical_and(flag, check)
                # If the argument check is False, and the sample is not empty
                # that's an error in the semantics of assess.
                return (
                    Masked.maybe(
                        jnp.logical_or(check_empty, check_nonempty), check * score
                    ),
                    Masked.maybe(check_nonempty, retval),
                )
            case _:
                # If the argument check is False, and the sample is not empty
                # that's an error in the semantics of assess.
                score, retval = self.gen_fn.assess(key, sample, tuple(inner_args))
                return (
                    Masked.maybe(check, score),
                    Masked.maybe(check, retval),
                )

    def importance_edit(
        self,
        key: PRNGKey,
        constraint: C,
        arguments: A,
    ) -> tuple[MaskedTrace, Weight, P]:
        raise NotImplementedError

    def project_edit(
        self,
        key: PRNGKey,
        trace: MaskedTrace,
        projection: P,
    ) -> tuple[Weight, C]:
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        trace: MaskedTrace,
        request: U | MaskedEditRequest[U],
        arguments: A,
    ) -> tuple[MaskedTrace, Weight, Retdiff, U | MaskedEditRequest[U]]:
        (flag_argument, *rest_args) = arguments
        match request:
            case _:
                # Assume trace was valid before, and is valid now -- so just compute the update.
                # We have a weight for valid -> valid, let's correct it after.
                new_inner_tr, weight, retdiff, bwd = request.edit(
                    key, trace.inner, tuple(rest_args)
                )

                # If the trace was invalid before, and is now valid -- then this is
                # equivalent from a move from the empty target, to the final target
                # defined by subrequest.
                weight_correction = (
                    weight
                    + jnp.logical_and(flag_argument, jnp.logical_not(trace.check))
                    * new_inner_tr.get_score()
                )

                # If the trace was valid before, and is now invalid -- then this is
                # equivalent to a move to the empty target.
                weight_correction = (
                    -1.0 * jnp.logical_not(flag_argument)
                ) * weight_correction

                return (
                    MaskedTrace(self, new_inner_tr, flag_argument),
                    weight_correction,
                    Masked(Diff.unknown_change(flag_argument), retdiff),
                    MaskedEditRequest(flag_argument, bwd),
                )


#############
# Decorator #
#############


def mask(f: GenerativeFunction) -> GenerativeFunction:
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
