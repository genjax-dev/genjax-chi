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


from genjax._src.core.generative import (
    Arguments,
    ChoiceMap,
    ChoiceMapConstraint,
    ChoiceMapProjection,
    ChoiceMapSample,
    EditRequest,
    GenerativeFunction,
    Masked,
    MaskedEditRequest,
    MaskedSample,
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
        return Masked(self.check, self.inner.get_retval())

    def get_score(self):
        return self.check * self.inner.get_score()


@Pytree.dataclass
class MaskedCombinator(
    Generic[Tr, A, R, U],
    GenerativeFunction[
        tuple[bool | BoolArray, A],
        ChoiceMapSample,
        Masked[R],
        ChoiceMapConstraint,
        SelectionProjection | ChoiceMapProjection,
        U,
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
    ) -> MaskedTrace:
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

    def edit(
        self,
        key: PRNGKey,
        trace: MaskedTrace,
        request: U | MaskedEditRequest[U],
        args: A,
    ) -> tuple[MaskedTrace, Weight, Retdiff, U | MaskedEditRequest[U]]:
        raise NotImplementedError


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
