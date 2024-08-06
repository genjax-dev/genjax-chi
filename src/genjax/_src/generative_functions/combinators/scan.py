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
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Arguments,
    ChoiceMap,
    ChoiceMapConstraint,
    ChoiceMapProjection,
    ChoiceMapSample,
    Constraint,
    EditRequest,
    EmptyConstraint,
    GenerativeFunction,
    MaskedConstraint,
    Retdiff,
    Sample,
    SampleCoercableToChoiceMap,
    Score,
    Selection,
    SelectionProjection,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import (
    ChoiceMapEditRequest,
    SelectionRegenerateRequest,
)
from genjax._src.core.interpreters.incremental import ChangeTangent, Diff, UnknownChange
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Callable,
    Generic,
    IntArray,
    Optional,
    PRNGKey,
    TypeVar,
    overload,
)

C = TypeVar("C", bound=Constraint)
Ca = TypeVar("Ca")
Sc1 = TypeVar("Sc1")
Sc2 = TypeVar("Sc2")
S = TypeVar("S", bound=Sample)
G = TypeVar("G", bound=GenerativeFunction)
Tr = TypeVar("Tr", bound=Trace)
A = TypeVar("A", bound=Arguments)
_Tr = TypeVar("_Tr", bound=Trace)
U = TypeVar("U", bound=EditRequest)


@Pytree.dataclass
class ScanTrace(
    Generic[G, Tr, Ca, Sc1, Sc2, S],
    SampleCoercableToChoiceMap,
    Trace[
        "ScanCombinator[G, Tr, Ca, Sc1, Sc2, S]",
        tuple[Ca, Sc1],
        ChoiceMapSample,
        tuple[Ca, Sc2],
    ],
):
    scan_gen_fn: "ScanCombinator[G, Tr, Ca, Sc1, Sc2, S]"
    inner: Tr
    arguments: tuple[Ca, Sc1]
    retval: tuple[Ca, Sc2]
    score: Score

    def get_args(self) -> tuple[Ca, Sc1]:
        return self.arguments

    def get_retval(self) -> tuple[Ca, Sc2]:
        return self.retval

    def get_sample(self) -> ChoiceMapSample:
        return ChoiceMapSample(
            jax.vmap(
                lambda idx, subtrace: ChoiceMap.idx(idx, subtrace.get_sample()),
            )(jnp.arange(self.scan_gen_fn.length), self.inner)
        )

    def get_choices(self) -> ChoiceMap:
        assert isinstance(self.inner, SampleCoercableToChoiceMap)
        return jax.vmap(
            lambda idx, subtrace: ChoiceMap.idx(idx, subtrace.get_choices())
        )(jnp.arange(self.scan_gen_fn.length), self.inner)

    def get_gen_fn(self) -> "ScanCombinator[G, Tr, Ca, Sc1, Sc2, S]":
        return self.scan_gen_fn

    def get_score(self) -> Score:
        return self.score


##########################
# Custom update requests #
##########################


@Pytree.dataclass
class IndexTangent(ChangeTangent):
    idx: IntArray
    tangent: ChangeTangent


@Pytree.dataclass(match_args=True)
class IndexEditRequest(
    Generic[U],
    EditRequest,
):
    index: IntArray
    subrequest: U
    validate: bool = Pytree.static(default=True)

    def edit(
        self,
        key: PRNGKey,
        trace: Trace,
        arguments: Arguments,
    ) -> tuple[Trace, Weight, Retdiff, "EditRequest"]:
        if self.validate:
            (carry_in, scanned_in) = arguments
            assert Diff.static_check_no_change(carry_in), Diff.tangent(carry_in)
        gen_fn = trace.get_gen_fn()
        return gen_fn.edit(key, trace, self, arguments)


###################
# Scan combinator #
###################


@Pytree.dataclass
class ScanCombinator(
    Generic[G, Tr, Ca, Sc1, Sc2, S, U],
    GenerativeFunction[
        ScanTrace[G, Tr, Ca, Sc1, Sc2, S],
        tuple[Ca, Sc1],
        ChoiceMapSample,
        tuple[Ca, Sc2],
        ChoiceMapConstraint,
        SelectionProjection | ChoiceMapProjection,
        ChoiceMapEditRequest | IndexEditRequest | SelectionRegenerateRequest,
    ],
):
    """`ScanCombinator` wraps a `kernel_gen_fn` [`genjax.GenerativeFunction`][]
    of type `(c, a) -> (c, b)` in a new [`genjax.GenerativeFunction`][] of type
    `(c, [a]) -> (c, [b])`, where.

    - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - `a` may be a primitive, an array type or a pytree (container) type with array leaves
    - `b` may be a primitive, an array type or a pytree (container) type with array leaves.

    The values traced by each call to the original generative function will be nested under an integer index that matches the loop iteration index that generated it.

    For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

    When the type of `xs` in the snippet below (denoted `[a]` above) is an array type or None, and the type of `ys` in the snippet below (denoted `[b]` above) is an array type, the semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

    ```python
    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)
    ```

    Unlike that Python version, both `xs` and `ys` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays. `None` is actually a special case of this, as it represents an empty pytree.

    The loop-carried value `c` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

    Attributes:
        kernel_gen_fn: a generative function to be scanned of type `(c, a) -> (c, b)`, meaning that `f` accepts two arguments where the first is a value of the loop carry and the second is a slice of `xs` along its leading axis, and that `f` returns a pair where the first element represents a new value for the loop carry and the second represents a slice of the output.

        length: optional integer specifying the number of loop iterations, which (if supplied) must agree with the sizes of leading axes of the arrays in the returned function's second argument. If supplied then the returned generative function can take `None` as its second argument.

        reverse: optional boolean specifying whether to run the scan iteration forward (the default) or in reverse, equivalent to reversing the leading axes of the arrays in both `xs` and in `ys`.

        unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many scan iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

    Examples:
        Use the [`genjax.GenerativeFunction.scan`][] method:
        ```python exec="yes" html="true" source="material-block" session="scan"
        import jax
        import genjax


        # A kernel_gen_fn generative function.
        @genjax.gen
        def random_walk_step(prev, _):
            x = genjax.normal(prev, 1.0) @ "x"
            return x, None


        init = 0.5
        key = jax.random.PRNGKey(314159)

        random_walk = random_walk_step.scan(n=1000)

        tr = jax.jit(random_walk.simulate)(key, (init, None))
        print(tr.render_html())
        ```

        Or use the [`genjax.scan`][] decorator:
        ```python exec="yes" html="true" source="material-block" session="scan"
        @genjax.scan(n=1000)
        @genjax.gen
        def random_walk(prev, _):
            x = genjax.normal(prev, 1.0) @ "x"
            return x, None


        tr = jax.jit(random_walk.simulate)(key, (init, None))
        print(tr.render_html())
        ```

    """

    kernel_gen_fn: GenerativeFunction[
        Tr,
        tuple[Ca, Sc1],
        S,
        tuple[Ca, Sc2],
        ChoiceMapConstraint,
        SelectionProjection | ChoiceMapProjection,
        ChoiceMapEditRequest,
    ]

    # Only required for `None` carry inputs
    length: Optional[int] = Pytree.static()
    reverse: bool = Pytree.static(default=False)
    unroll: int | bool = Pytree.static(default=1)

    # To get the type of return value, just invoke
    # the scanned over source (with abstract tracer arguments).
    def __abstract_call__(self, carry_in: Ca, scanned_in: Sc1) -> tuple[Ca, Sc2]:  # type:ignore
        def _inner(
            carry_in: Ca,
            scanned_in: Sc1,
        ):
            v, scanned_out = self.kernel_gen_fn.__abstract_call__(carry_in, scanned_in)
            return v, scanned_out

        v, scanned_out = jax.lax.scan(
            _inner,
            carry_in,
            scanned_in,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )
        return v, scanned_out

    def simulate(
        self,
        key: PRNGKey,
        arguments: tuple[Ca, Sc1],
    ) -> ScanTrace[G, Tr, Ca, Sc1, Sc2, S]:
        carry, scanned_in = arguments

        def _inner_simulate(key, carry, scanned_in):
            tr = self.kernel_gen_fn.simulate(key, (carry, scanned_in))
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out)

        def _inner(carry, scanned_over):
            key, count, carried_value = carry
            key = jax.random.fold_in(key, count)
            (carry, score), (tr, scanned_out) = _inner_simulate(
                key, carried_value, scanned_over
            )

            return (key, count + 1, carry), (tr, scanned_out, score)

        (_, _, carried_out), (tr, scanned_out, scores) = jax.lax.scan(
            _inner,
            (key, 0, carry),
            scanned_in,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )

        return ScanTrace(
            self,  # type:ignore
            tr,
            arguments,
            (carried_out, scanned_out),
            jnp.sum(scores),
        )  # type:ignore

    def assess(
        self,
        key: PRNGKey,
        sample: ChoiceMapSample,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[Score, tuple[Ca, Sc2]]:
        (carry_in, else_to_scan) = arguments

        def _assess(carry, scanned_in):
            key, idx, carry_in = carry
            (sample, scanned_in) = scanned_in
            subsample = sample.get_submap(idx)
            sub_key = jax.random.fold_in(key, idx)
            score, retval = self.kernel_gen_fn.assess(
                sub_key, subsample, (carry_in, scanned_in)
            )
            (carry_out, scanned_out) = retval
            idx += 1
            return (key, idx, carry_out), (scanned_out, score)

        (_, _, carry_out), (scanned_out, scores) = jax.lax.scan(
            _assess,
            (key, 0, carry_in),
            (sample, else_to_scan),
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )
        return (
            jnp.sum(scores),
            (carry_out, scanned_out),
        )

    def importance_edit(
        self,
        key: PRNGKey,
        constraint: ChoiceMapConstraint,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[ScanTrace[G, Tr, Ca, Sc1, Sc2, S], Weight, ChoiceMapProjection]:
        (carry, scanned_in) = arguments

        def _inner_importance(
            key: PRNGKey,
            constraint: ChoiceMapConstraint,
            carry: Ca,
            scanned_in: Sc1,
        ):
            tr, w, bwd_projection = self.kernel_gen_fn.importance_edit(
                key,
                constraint,
                (carry, scanned_in),
            )
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out, w, bwd_projection)

        def _get_subconstraint(
            constraint: ChoiceMapConstraint,
            idx: IntArray,
        ):
            match constraint:
                case EmptyConstraint():
                    return constraint
                case ChoiceMapConstraint():
                    return constraint.get_submap(idx)
                case MaskedConstraint(flag, subconstraint):
                    sub = _get_subconstraint(subconstraint, idx)
                    return MaskedConstraint(flag, sub)

        def _importance(carry, scanned_over):
            key, idx, carried_value = carry
            key = jax.random.fold_in(key, idx)
            subconstraint = _get_subconstraint(constraint, idx)
            (carry, score), (tr, scanned_out, w, inner_projection) = _inner_importance(
                key, subconstraint, carried_value, scanned_over
            )
            bwd_projection = ChoiceMapProjection(ChoiceMap.idx(idx, inner_projection))

            return (key, idx + 1, carry), (tr, scanned_out, score, w, bwd_projection)

        (_, _, carried_out), (tr, scanned_out, scores, ws, bwd_projections) = (
            jax.lax.scan(
                _importance,
                (key, 0, carry),
                scanned_in,
                length=self.length,
                reverse=self.reverse,
                unroll=self.unroll,
            )
        )
        return (
            ScanTrace(
                self,  # type:ignore
                tr,
                arguments,
                (carried_out, scanned_out),
                jnp.sum(scores),
            ),
            jnp.sum(ws),
            bwd_projections,
        )  # type:ignore

    def project_edit(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        projection: SelectionProjection | ChoiceMapProjection,
    ) -> tuple[Weight, ChoiceMapConstraint]:
        def _inner_project(
            key: PRNGKey,
            inner_trace: Tr,
            projection: SelectionProjection | ChoiceMapProjection,
        ):
            w, constraint = self.kernel_gen_fn.project_edit(
                key,
                inner_trace,
                projection,
            )
            return w, constraint

        def _get_subprojection(
            projection: SelectionProjection | ChoiceMapProjection,
            idx: IntArray,
        ) -> SelectionProjection | ChoiceMapProjection:
            return projection(idx)  # type: ignore

        def _project(carry, scanned):
            key, idx = carry
            inner_trace = scanned
            key = jax.random.fold_in(key, idx)
            subprojection = _get_subprojection(projection, idx)
            (w, bwd_constraint) = _inner_project(
                key,
                inner_trace,
                subprojection,
            )

            return (key, idx + 1), (w, bwd_constraint)

        (_, _), (ws, bwd_constraints) = jax.lax.scan(
            _project,
            (key, 0),
            trace.inner,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )
        return (
            jnp.sum(ws),
            bwd_constraints,
        )  # type:ignore

    def choice_map_edit(
        self,
        key: PRNGKey,
        trace: ScanTrace[G, Tr, Ca, Sc1, Sc2, S],
        constraint: ChoiceMapConstraint,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[ScanTrace[G, Tr, Ca, Sc1, Sc2, S], Weight, ChoiceMapConstraint]:
        def _inner_update(
            key: PRNGKey,
            subtrace: Tr,
            subconstraint: ChoiceMapConstraint,
            carry_in: Ca,
            scanned_in: Sc1,
        ) -> tuple[
            tuple[Ca, Score],
            tuple[Tr, Sc2, Weight, Constraint],
        ]:
            request = ChoiceMapEditRequest(subconstraint)
            new_subtrace, w, retdiff, bwd_move = request.edit(
                key, subtrace, (carry_in, scanned_in)
            )
            score = new_subtrace.get_score()
            carry_out, scanned_out = new_subtrace.get_retval()
            bwd_constraint = bwd_move.constraint
            return (carry_out, score), (
                new_subtrace,
                scanned_out,
                w,
                bwd_constraint,
            )  # type: ignore

        def _get_subconstraint(
            constraint: ChoiceMapConstraint,
            idx: IntArray,
        ) -> ChoiceMapConstraint:
            return constraint.get_submap(idx)

        def _update(
            carry: tuple[PRNGKey, IntArray, Ca],
            scanned_over: tuple[Tr, Sc1],
        ) -> tuple[
            tuple[PRNGKey, IntArray, Ca],
            tuple[Tr, Sc2, Score, Weight, ChoiceMap],
        ]:
            key, idx, carried_value = carry
            subtrace, scanned_in = scanned_over
            key = jax.random.fold_in(key, idx)
            subconstraint = _get_subconstraint(constraint, idx)
            (
                (carry_out, score),
                (new_subtrace, scanned_out, w, bwd_constraint),
            ) = _inner_update(
                key,
                subtrace,
                subconstraint,
                carried_value,
                scanned_in,
            )
            bwd_constraint = ChoiceMap.idx(idx, bwd_constraint)

            return (key, idx + 1, carry_out), (
                new_subtrace,
                scanned_out,
                score,
                w,
                bwd_constraint,
            )

        (carry_in, scanned_in) = arguments
        (
            (_, _, carry_out),
            (new_subtraces, scanned_out, scores, ws, bwd_chm),
        ) = jax.lax.scan(
            _update,
            (key, jnp.array(0), carry_in),
            (trace.inner, scanned_in),
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )
        bwd_constraint = ChoiceMapConstraint(bwd_chm)
        return (
            ScanTrace(
                self,
                new_subtraces,
                arguments,
                (carry_out, scanned_out),
                jnp.sum(scores),
            ),
            jnp.sum(ws),
            bwd_constraint,
        )

    def selection_regenerate_edit(
        self,
        key: PRNGKey,
        trace: ScanTrace[G, Tr, Ca, Sc1, Sc2, S],
        selection: Selection,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[ScanTrace[G, Tr, Ca, Sc1, Sc2, S], Weight, ChoiceMapConstraint]:
        raise NotImplementedError

    def index_edit(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        idx: IntArray,
        subrequest: EditRequest,
        validate: bool,
    ) -> tuple[ScanTrace, Weight, Retdiff, EditRequest]:
        starting_subslice = jtu.tree_map(lambda v: v[idx], trace.inner)
        affected_subslice = jtu.tree_map(lambda v: v[idx + 1], trace.inner)
        starting_args = starting_subslice.get_args()
        (
            updated_start,
            start_w,
            starting_retdiff,
            bwd_request,
        ) = subrequest.edit(key, starting_subslice, starting_args)
        (carry_retdiff, _) = starting_retdiff
        _, scanned_in = trace.get_args()
        next_slice_argdiffs = carry_retdiff, Diff.no_change(scanned_in[idx + 1])
        next_slice_args = Diff.tree_primal(next_slice_argdiffs)
        request = ChoiceMapEditRequest(ChoiceMapConstraint(ChoiceMap.empty()))
        (
            updated_end,
            end_w,
            (ending_carry_diff, ending_scanned_out_diff),
            _,
        ) = request.edit(key, affected_subslice, next_slice_args)

        if validate:
            # Must be true for this type of update to be valid.
            assert Diff.static_check_no_change(ending_carry_diff)

        def _mutate_in_place(arr, updated_start, updated_end):
            arr = arr.at[idx].set(updated_start)
            arr = arr.at[idx + 1].set(updated_end)
            return arr

        new_inner = jtu.tree_map(
            _mutate_in_place, trace.inner, updated_start, updated_end
        )
        (carry_out, scanned_out) = new_inner.get_retval()

        scanned_out_retdiff = Diff(
            scanned_out,
            IndexTangent(idx, UnknownChange),
        )
        carry_out_retdiff = Diff.unknown_change(carry_out)

        return (
            ScanTrace(
                self,
                new_inner,
                new_inner.get_args(),
                new_inner.get_retval(),
                jnp.sum(new_inner.get_score()),
            ),
            start_w + end_w,
            (carry_out_retdiff, scanned_out_retdiff),
            bwd_request,
        )

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        request: ChoiceMapEditRequest,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[ScanTrace, Weight, Retdiff, ChoiceMapEditRequest]:
        pass

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        request: IndexEditRequest,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[ScanTrace, Weight, Retdiff, IndexEditRequest]:
        pass

    @overload
    def edit(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        request: SelectionRegenerateRequest,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[ScanTrace, Weight, Retdiff, ChoiceMapEditRequest]:
        pass

    def edit(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        request: ChoiceMapEditRequest | IndexEditRequest | SelectionRegenerateRequest,
        arguments: tuple[Ca, Sc1],
    ) -> tuple[
        ScanTrace,
        Weight,
        Retdiff,
        ChoiceMapEditRequest | IndexEditRequest | SelectionRegenerateRequest,
    ]:
        match request:
            case ChoiceMapEditRequest(choice_map_constraint):
                arguments = Diff.primal(arguments)
                new_trace, weight, bwd_constraint = self.choice_map_edit(
                    key, trace, choice_map_constraint, arguments
                )
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    ChoiceMapEditRequest(bwd_constraint),
                )
            case IndexEditRequest(index, subrequest, validate):
                new_trace, weight, retdiff, bwd_request = self.index_edit(
                    key, trace, index, subrequest, validate
                )
                return (
                    new_trace,
                    weight,
                    retdiff,
                    IndexEditRequest(index, bwd_request, validate),
                )

            case SelectionRegenerateRequest(selection):
                new_trace, weight, bwd_constraint = self.selection_regenerate_edit(
                    key, trace, selection, arguments
                )
                return (
                    new_trace,
                    weight,
                    Diff.unknown_change(new_trace.get_retval()),
                    ChoiceMapEditRequest(bwd_constraint),
                )


##############
# Decorators #
##############


def scan(
    *,
    n: Optional[int] = None,
    reverse: bool = False,
    unroll: int | bool = 1,
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type
    `(c, a) -> (c, b)`and returns a new [`genjax.GenerativeFunction`][] of type
    `(c, [a]) -> (c, [b])` where.

    - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - `a` may be a primitive, an array type or a pytree (container) type with array leaves
    - `b` may be a primitive, an array type or a pytree (container) type with array leaves.

    The values traced by each call to the original generative function will be nested under an integer index that matches the loop iteration index that generated it.

    For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

    When the type of `xs` in the snippet below (denoted `[a]` above) is an array type or None, and the type of `ys` in the snippet below (denoted `[b]` above) is an array type, the semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

    ```python
    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = f(carry, x)
            ys.append(y)
        return carry, np.stack(ys)
    ```

    Unlike that Python version, both `xs` and `ys` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays. `None` is actually a special case of this, as it represents an empty pytree.

    The loop-carried value `c` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

    Args:
        n: optional integer specifying the number of loop iterations, which (if supplied) must agree with the sizes of leading axes of the arrays in the returned function's second argument. If supplied then the returned generative function can take `None` as its second argument.

        reverse: optional boolean specifying whether to run the scan iteration forward (the default) or in reverse, equivalent to reversing the leading axes of the arrays in both `xs` and in `ys`.

        unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many scan iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

    Returns:
        A new [`genjax.GenerativeFunction`][] that takes a loop-carried value and a new input, and returns a new loop-carried value along with either `None` or an output to be collected into the second return value.

    Examples:
        Scan for 1000 iterations with no array input:
        ```python exec="yes" html="true" source="material-block" session="scan"
        import jax
        import genjax


        @genjax.scan(n=1000)
        @genjax.gen
        def random_walk(prev, _):
            x = genjax.normal(prev, 1.0) @ "x"
            return x, None


        init = 0.5
        key = jax.random.PRNGKey(314159)

        tr = jax.jit(random_walk.simulate)(key, (init, None))
        print(tr.render_html())
        ```

        Scan across an input array:
        ```python exec="yes" html="true" source="material-block" session="scan"
        import jax.numpy as jnp


        @genjax.scan()
        @genjax.gen
        def add_and_square_all(sum, x):
            new_sum = sum + x
            return new_sum, sum * sum


        init = 0.0
        xs = jnp.ones(10)

        tr = jax.jit(add_and_square_all.simulate)(key, (init, xs))

        # The retval has the final carry and an array of all `sum*sum` returned.
        print(tr.render_html())
        ```

    """

    def decorator(f) -> ScanCombinator:
        return ScanCombinator(
            f,
            length=n,
            reverse=reverse,
            unroll=unroll,
        )

    return decorator


def prepend_initial_acc(arguments, ret):
    """Prepends the initial accumulator value to the array of accumulated
    values.

    This function is used in the context of scan operations to include the initial
    accumulator state in the output, effectively providing a complete history of
    the accumulator's values throughout the scan.

    Args:
        arguments: A tuple containing the initial arguments to the scan operation. The first element is expected to be the initial accumulator value.
        ret: A tuple containing the final accumulator value and an array of intermediate accumulator values from the scan operation.

    Returns:
        A tree structure where each leaf is an array with the initial accumulator value prepended to the corresponding array of intermediate values.

    Note:
        This function uses JAX's tree mapping to handle nested structures in the accumulator, allowing it to work with complex accumulator types.

    """
    init_acc = arguments[0]
    xs = ret[1]

    def cat(init, arr):
        return jnp.concatenate([jnp.array(init)[jnp.newaxis], arr])

    return jax.tree.map(cat, init_acc, xs)


def accumulate(
    *, reverse: bool = False, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type
    `(c, a) -> c` and returns a new [`genjax.GenerativeFunction`][] of type
    `(c, [a]) -> [c]` where.

    - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - `[c]` is an array of all loop-carried values seen during iteration (including the first)
    - `a` may be a primitive, an array type or a pytree (container) type with array leaves

    All traced values are nested under an index.

    For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

    The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation (note the similarity to [`itertools.accumulate`](https://docs.python.org/3/library/itertools.html#itertools.accumulate)):

    ```python
    def accumulate(f, init, xs):
        carry = init
        carries = [init]
        for x in xs:
            carry = f(carry, x)
            carries.append(carry)
        return carries
    ```

    Unlike that Python version, both `xs` and `carries` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays.

    The loop-carried value `c` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

    Args:
        reverse: optional boolean specifying whether to run the accumulation forward (the default) or in reverse, equivalent to reversing the leading axes of the arrays in both `xs` and in `carries`.

        unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

    Examples:
        accumulate a running total:
        ```python exec="yes" html="true" source="material-block" session="scan"
        import jax
        import genjax
        import jax.numpy as jnp


        @genjax.accumulate()
        @genjax.gen
        def add(sum, x):
            new_sum = sum + x
            return new_sum


        init = 0.0
        key = jax.random.PRNGKey(314159)
        xs = jnp.ones(10)

        tr = jax.jit(add.simulate)(key, (init, xs))
        print(tr.render_html())
        ```

    """

    def decorator(f: GenerativeFunction):
        return (
            f.map(lambda ret: (ret, ret))
            .scan(reverse=reverse, unroll=unroll)
            .dimap(pre=lambda *arguments: arguments, post=prepend_initial_acc)
        )

    return decorator


def reduce(
    *, reverse: bool = False, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type
    `(c, a) -> c` and returns a new [`genjax.GenerativeFunction`][] of type
    `(c, [a]) -> c` where.

    - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - `a` may be a primitive, an array type or a pytree (container) type with array leaves

    All traced values are nested under an index.

    For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

    The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation (note the similarity to [`functools.reduce`](https://docs.python.org/3/library/itertools.html#functools.reduce)):

    ```python
    def reduce(f, init, xs):
        carry = init
        for x in xs:
            carry = f(carry, x)
        return carry
    ```

    Unlike that Python version, both `xs` and `carry` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays.

    The loop-carried value `c` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

    Args:
        reverse: optional boolean specifying whether to run the accumulation forward (the default) or in reverse, equivalent to reversing the leading axis of the array `xs`.

        unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

    Examples:
        sum an array of numbers:
        ```python exec="yes" html="true" source="material-block" session="scan"
        import jax
        import genjax
        import jax.numpy as jnp


        @genjax.reduce()
        @genjax.gen
        def add(sum, x):
            new_sum = sum + x
            return new_sum


        init = 0.0
        key = jax.random.PRNGKey(314159)
        xs = jnp.ones(10)

        tr = jax.jit(add.simulate)(key, (init, xs))
        print(tr.render_html())
        ```

    """

    def decorator(f: GenerativeFunction):
        return (
            f.map(lambda ret: (ret, None))
            .scan(reverse=reverse, unroll=unroll)
            .map(lambda ret: ret[0])
        )

    return decorator


def iterate(
    *, n: int, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type
    `a -> a` and returns a new [`genjax.GenerativeFunction`][] of type `a ->
    [a]` where.

    - `a` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - `[a]` is an array of all `a`, `f(a)`, `f(f(a))` etc. values seen during iteration.

    All traced values are nested under an index.

    The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

    ```python
    def iterate(f, n, init):
        input = init
        seen = [init]
        for _ in range(n):
            input = f(input)
            seen.append(input)
        return seen
    ```

    `init` may be an arbitrary pytree value, and so multiple arrays can be iterated over at once and produce multiple output arrays.

    The iterated value `a` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `a` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

    Args:
        n: the number of iterations to run.

        unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

    Examples:
        iterative addition, returning all intermediate sums:
        ```python exec="yes" html="true" source="material-block" session="scan"
        import jax
        import genjax


        @genjax.iterate(n=100)
        @genjax.gen
        def inc(x):
            return x + 1


        init = 0.0
        key = jax.random.PRNGKey(314159)

        tr = jax.jit(inc.simulate)(key, (init,))
        print(tr.render_html())
        ```

    """

    def decorator(f: GenerativeFunction):
        # strip off the JAX-supplied `None` on the way in, accumulate `ret` on the way out.
        return (
            f.dimap(
                pre=lambda *arguments: arguments[:-1], post=lambda _, ret: (ret, ret)
            )
            .scan(n=n, unroll=unroll)
            .dimap(pre=lambda *arguments: (*arguments, None), post=prepend_initial_acc)
        )

    return decorator


def iterate_final(
    *,
    n: int,
    unroll: int | bool = 1,
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type
    `a -> a` and returns a new [`genjax.GenerativeFunction`][] of type `a -> a`
    where.

    - `a` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - the original function is invoked `n` times with each input coming from the previous invocation's output, so that the new function returns $f^n(a)$

    All traced values are nested under an index.

    The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

    ```python
    def iterate_final(f, n, init):
        ret = init
        for _ in range(n):
            ret = f(ret)
        return ret
    ```

    `init` may be an arbitrary pytree value, and so multiple arrays can be iterated over at once and produce multiple output arrays.

    The iterated value `a` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `a` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

    Args:
        n: the number of iterations to run.

        unroll: optional positive int or bool specifying, in the underlying operation of the scan primitive, how many iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is competely unrolled (i.e. `unroll=True`) or left completely unrolled (i.e. `unroll=False`).

    Examples:
        iterative addition:
        ```python exec="yes" html="true" source="material-block" session="scan"
        import jax
        import genjax


        @genjax.iterate_final(n=100)
        @genjax.gen
        def inc(x):
            return x + 1


        init = 0.0
        key = jax.random.PRNGKey(314159)

        tr = jax.jit(inc.simulate)(key, (init,))
        print(tr.render_html())
        ```

    """

    def decorator(f: GenerativeFunction):
        # strip off the JAX-supplied `None` on the way in, no accumulation on the way out.
        return (
            f.dimap(
                pre=lambda *arguments: arguments[:-1], post=lambda _, ret: (ret, None)
            )
            .scan(n=n, unroll=unroll)
            .dimap(
                pre=lambda *arguments: (*arguments, None), post=lambda _, ret: ret[0]
            )
        )

    return decorator
