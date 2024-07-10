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
    Argdiffs,
    ChoiceMap,
    ChoiceMapUpdateRequest,
    Constraint,
    EmptyTrace,
    EmptyUpdateRequest,
    GenerativeFunction,
    ImportanceUpdateRequest,
    IncrementalUpdateRequest,
    Retdiff,
    Sample,
    Score,
    Selection,
    Trace,
    UpdateRequest,
    Weight,
)
from genjax._src.core.generative.choice_map import ChoiceMapConstraint
from genjax._src.core.generative.core import Arguments
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Int,
    IntArray,
    Optional,
    PRNGKey,
    Tuple,
    dispatch,
    overload,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class ScanTrace(Trace):
    scan_gen_fn: "ScanCombinator"
    inner: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def get_args(self) -> Tuple:
        return self.args

    def get_retval(self):
        return self.retval

    def get_sample(self):
        return jax.vmap(
            lambda idx, subtrace: ChoiceMap.idx(idx, subtrace.get_sample()),
        )(jnp.arange(self.scan_gen_fn.length), self.inner)

    def get_gen_fn(self):
        return self.scan_gen_fn

    def get_score(self):
        return self.score


#######################
# Custom update specs #
#######################


@Pytree.dataclass(match_args=True)
class IndexUpdateRequest(UpdateRequest):
    index: IntArray
    subrequest: UpdateRequest


@Pytree.dataclass(match_args=True)
class StaticExtensionUpdateRequest(UpdateRequest):
    """
    Denotes an extension of a `ScanTrace` by static `extension_length` number of steps.
    """

    constraint: Constraint
    new_scans: Any
    extension_length: Int = Pytree.static()


@Pytree.dataclass(match_args=True)
class StaticRetractionUpdateRequest(UpdateRequest):
    retraction_length: Int = Pytree.static()


###################
# Scan combinator #
###################


@Pytree.dataclass
class ScanCombinator(GenerativeFunction):
    """
    `ScanCombinator` wraps a `kernel_gen_fn` [`genjax.GenerativeFunction`][] of type `(c, a) -> (c, b)` in a new [`genjax.GenerativeFunction`][] of type `(c, [a]) -> (c, [b])`, where

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

    kernel_gen_fn: GenerativeFunction

    # Only required for `None` carry inputs
    length: Optional[Int] = Pytree.static()
    reverse: bool = Pytree.static(default=False)
    unroll: int | bool = Pytree.static(default=1)

    # To get the type of return value, just invoke
    # the scanned over source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        (carry, scanned_in) = args

        def _inner(carry, scanned_in):
            v, scanned_out = self.kernel_gen_fn.__abstract_call__(carry, scanned_in)
            return v, scanned_out

        v, scanned_out = jax.lax.scan(
            _inner,
            carry,
            scanned_in,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )

        return v, scanned_out

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> ScanTrace:
        carry, scanned_in = args

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

        return ScanTrace(self, tr, args, (carried_out, scanned_out), jnp.sum(scores))

    @typecheck
    def update_importance(
        self,
        key: PRNGKey,
        trace: Trace,
        constraint: ChoiceMapConstraint,
        args: Arguments,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        (carry, scanned_in) = args

        def _inner_importance(key, constraint, carry, scanned_in):
            tr, w, _retdiff, bwd_request = self.kernel_gen_fn.update(
                key,
                EmptyTrace(self.kernel_gen_fn),
                ImportanceUpdateRequest((carry, scanned_in), constraint),
            )
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out, w, bwd_request)

        def _importance(carry, scanned_over):
            key, idx, carried_value = carry
            key = jax.random.fold_in(key, idx)
            submap = constraint.get_submap(idx)
            (carry, score), (tr, scanned_out, w, inner_bwd_request) = _inner_importance(
                key, submap, carried_value, scanned_over
            )
            bwd_request = ChoiceMap.idx(idx, inner_bwd_request)

            return (key, idx + 1, carry), (tr, scanned_out, score, w, bwd_request)

        (_, _, carried_out), (tr, scanned_out, scores, ws, bwd_requests) = jax.lax.scan(
            _importance,
            (key, 0, carry),
            scanned_in,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )
        bwd_request = ChoiceMapUpdateRequest(bwd_requests)
        return (
            ScanTrace(self, tr, args, (carried_out, scanned_out), jnp.sum(scores)),
            jnp.sum(ws),
            Diff.unknown_change((carried_out, scanned_out)),
            bwd_request,
        )

    def update_index(
        self,
        key: PRNGKey,
        trace: Trace,
        index: IntArray,
        request: UpdateRequest,
    ):
        starting_subslice = jtu.tree_map(lambda v: v[index], trace.inner)
        affected_subslice = jtu.tree_map(lambda v: v[index + 1], trace.inner)
        starting_argdiffs = Diff.no_change(starting_subslice.get_args())
        (
            updated_start,
            start_w,
            starting_retdiff,
            bwd_request,
        ) = self.kernel_gen_fn.update(
            key,
            starting_subslice,
            request,
            starting_argdiffs,
        )
        updated_end, end_w, ending_retdiff, _ = self.kernel_gen_fn.update(
            key,
            affected_subslice,
            EmptyUpdateRequest(),
            starting_retdiff,
        )

        # Must be true for this type of update to be valid.
        assert Diff.static_check_no_change(ending_retdiff)

        def _mutate_in_place(arr, updated_start, updated_end):
            arr = arr.at[index].set(updated_start)
            arr = arr.at[index + 1].set(updated_end)
            return arr

        new_inner = jtu.tree_map(
            _mutate_in_place, trace.inner, updated_start, updated_end
        )
        new_retvals = new_inner.get_retval()
        return (
            ScanTrace(
                self,
                new_inner,
                new_inner.get_args(),
                new_retvals,
                jnp.sum(new_inner.get_score()),
            ),
            start_w + end_w,
            Diff.unknown_change(new_retvals),
            IndexUpdateRequest(index, bwd_request),
        )

    @overload
    def update_incremental(
        self,
        key: PRNGKey,
        trace: Trace,
        request: IndexUpdateRequest,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        index, subrequest = request.index, request.subrequest
        if Diff.static_check_no_change(argdiffs):
            return self.update_index(key, trace, index, subrequest)
        else:
            choice_map_request = ChoiceMapUpdateRequest(
                ChoiceMap.idx(index, subrequest)
            )
            return self.update_incremental(key, trace, choice_map_request, argdiffs)

    def _get_subrequest(
        self,
        request: UpdateRequest,
        idx: IntArray,
    ) -> UpdateRequest:
        match request:
            case ChoiceMap():
                return request(idx)

            case Selection():
                subrequest = request(idx)
                return subrequest

            case _:
                raise Exception(f"Not implemented subrequest: {request}")

    @overload
    def update_incremental(
        self,
        key: PRNGKey,
        trace: Trace,
        request: UpdateRequest,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        carry_diff, *scanned_in_diff = Diff.tree_diff_unknown_change(
            Diff.tree_primal(argdiffs)
        )

        def _inner_update(key, subtrace, subrequest, carry, scanned_in):
            (
                new_subtrace,
                w,
                kernel_retdiff,
                bwd_request,
            ) = self.kernel_gen_fn.update(
                key,
                subtrace,
                IncrementalUpdateRequest(
                    (carry, scanned_in),
                    subrequest,
                ),
            )
            (carry_retdiff, scanned_out_retdiff) = kernel_retdiff
            score = new_subtrace.get_score()
            return (carry_retdiff, score), (
                new_subtrace,
                scanned_out_retdiff,
                w,
                bwd_request,
            )

        def _update(carry, scanned_over):
            key, idx, carried_value = carry
            (subtrace, *scanned_in) = scanned_over
            key = jax.random.fold_in(key, idx)
            subrequest = self._get_subrequest(request, idx)
            (
                (carry, score),
                (new_subtrace, scanned_out, w, inner_bwd_request),
            ) = _inner_update(key, subtrace, subrequest, carried_value, scanned_in)
            bwd_request = ChoiceMap.idx(idx, inner_bwd_request)

            return (key, idx + 1, carry), (
                new_subtrace,
                scanned_out,
                score,
                w,
                bwd_request,
            )

        (
            (_, _, carried_out_diff),
            (new_subtraces, scanned_out_diff, scores, ws, bwd_requests),
        ) = jax.lax.scan(
            _update,
            (key, 0, carry_diff),
            (trace.inner, *scanned_in_diff),
            length=self.max_length,
        )
        carried_out, scanned_out = Diff.tree_primal((
            carried_out_diff,
            scanned_out_diff,
        ))
        return (
            ScanTrace(
                self,
                new_subtraces,
                Diff.tree_primal(argdiffs),
                (carried_out, scanned_out),
                jnp.sum(scores),
            ),
            jnp.sum(ws),
            (carried_out_diff, scanned_out_diff),
            bwd_requests,
        )

    @dispatch
    def update_incremental(
        self, key, trace, request, argdiffs
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        raise NotImplementedError

    @overload
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        request: StaticExtensionUpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        extension_length, new_scans, constraint = (
            request.extension_length,
            request.new_scans,
            request.constraint,
        )
        constraint = ChoiceMapConstraint(
            jax.vmap(lambda idx, constraint: ChoiceMap.idx(idx, constraint))(
                jnp.arange(extension_length), constraint
            )
            if extension_length != 1
            else ChoiceMap.idx(1, constraint)
        )
        internal_combinator = ScanCombinator(self.kernel_gen_fn, extension_length)
        (old_carried, old_scanned_out) = trace.get_retval()
        extension_tr, w = internal_combinator.importance(
            key, constraint, (old_carried, new_scans)
        )
        (final_carried, new_scanned_out) = extension_tr.get_retval()
        final_inner_tr = Pytree.tree_concat([trace.inner, extension_tr.inner])
        score = jnp.sum(final_inner_tr.get_score())
        final_scanned_out = Pytree.tree_concat([old_scanned_out, new_scanned_out])
        full_combinator = ScanCombinator(
            self.kernel_gen_fn, self.max_length + extension_length
        )
        tr = ScanTrace(
            full_combinator,
            final_inner_tr,
            trace.get_args(),
            (final_carried, final_scanned_out),
            score,
        )
        return (
            tr,
            w,
            Diff.tree_diff_unknown_change((final_carried, new_scanned_out)),
            StaticRetractionUpdateRequest(extension_length),
        )

    @overload
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        request: StaticRetractionUpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        raise NotImplementedError

    @overload
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        request: ImportanceUpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        args, constraint = request.args, request.constraint
        return self.update_importance(key, trace, constraint, args)

    @overload
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        request: IncrementalUpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        argdiffs, subrequest = request.argdiffs, request.subrequest
        return self.update_incremental(key, trace, subrequest, argdiffs)

    @GenerativeFunction.gfi_boundary
    @dispatch
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        request: UpdateRequest,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateRequest]:
        raise NotImplementedError

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        (carry, scanned_in) = args
        assert isinstance(sample, ChoiceMap)

        def _inner_assess(sample, carry, scanned_in):
            score, retval = self.kernel_gen_fn.assess(sample, (carry, scanned_in))
            (carry, scanned_out) = retval
            return (carry, score), scanned_out

        def _assess(carry, scanned_over):
            idx, carried_value = carry
            submap = sample.get_submap(idx)
            (carry, score), scanned_out = _inner_assess(
                submap, carried_value, scanned_over
            )

            return (idx + 1, carry), (scanned_out, score)

        (_, carried_out), (scanned_out, scores) = jax.lax.scan(
            _assess,
            (0, carry),
            scanned_in,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )
        return (
            jnp.sum(scores),
            (carried_out, scanned_out),
        )

    @classmethod
    def index_update(
        cls,
        idx: IntArray,
        request: UpdateRequest,
    ) -> IndexUpdateRequest:
        return IndexUpdateRequest(idx, request)

    @classmethod
    def extension_update(
        cls,
        extension_length: Int,
        new_scans: Any,
        constraint: Constraint,
    ) -> StaticExtensionUpdateRequest:
        return StaticExtensionUpdateRequest(constraint, new_scans, extension_length)

    @classmethod
    def propose_extension_update(
        cls,
        extension_length: Int,
        new_scans: Any,
        proposal: GenerativeFunction,
    ):
        pass


##############
# Decorators #
##############


@typecheck
def scan(
    *, n: Optional[Int] = None, reverse: bool = False, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type `(c, a) -> (c, b)`and returns a new [`genjax.GenerativeFunction`][] of type `(c, [a]) -> (c, [b])` where

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

    def decorator(f):
        return ScanCombinator(f, length=n, reverse=reverse, unroll=unroll)

    return decorator


@typecheck
def accumulate(
    *, reverse: bool = False, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type `(c, a) -> c` and returns a new [`genjax.GenerativeFunction`][] of type `(c, [a]) -> [c]` where

    - `c` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - `[c]` is an array of all loop-carried values seen during iteration (excluding the first)
    - `a` may be a primitive, an array type or a pytree (container) type with array leaves

    All traced values are nested under an index.

    For any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

    The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation (note the similarity to [`itertools.accumulate`](https://docs.python.org/3/library/itertools.html#itertools.accumulate)):

    ```python
    def accumulate(f, init, xs):
        carry = init
        carries = []
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

    def decorator(f):
        return (
            f.map(lambda ret: (ret, ret))
            .scan(reverse=reverse, unroll=unroll)
            .map(lambda ret: ret[1])
        )

    return decorator


@typecheck
def reduce(
    *, reverse: bool = False, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type `(c, a) -> c` and returns a new [`genjax.GenerativeFunction`][] of type `(c, [a]) -> c` where

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


@typecheck
def iterate(
    *, n: Int, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type `a -> a` and returns a new [`genjax.GenerativeFunction`][] of type `a -> [a]` where

    - `a` is a loop-carried value, which must hold a fixed shape and dtype across all iterations
    - `[a]` is an array of all `f(a)`, `f(f(a))` etc. values seen during iteration (excluding tjhe initial `a`)

    All traced values are nested under an index.

    The semantics of the returned [`genjax.GenerativeFunction`][] are given roughly by this Python implementation:

    ```python
    def iterate(f, n, init):
        input = init
        seen = []
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
            f.dimap(pre=lambda *args: args[:-1], post=lambda _, ret: (ret, ret))
            .scan(n=n, unroll=unroll)
            .dimap(pre=lambda *args: (*args, None), post=lambda _, ret: ret[1])
        )

    return decorator


@typecheck
def iterate_final(
    *, n: Int, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] of type `a -> a` and returns a new [`genjax.GenerativeFunction`][] of type `a -> a` where

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
            f.dimap(pre=lambda *args: args[:-1], post=lambda _, ret: (ret, None))
            .scan(n=n, unroll=unroll)
            .dimap(pre=lambda *args: (*args, None), post=lambda _, ret: ret[0])
        )

    return decorator
