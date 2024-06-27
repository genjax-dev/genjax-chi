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
    EmptyProblem,
    EmptyTrace,
    GenerativeFunction,
    GenericProblem,
    ImportanceProblem,
    Retdiff,
    Sample,
    Score,
    Selection,
    Trace,
    UpdateProblem,
    Weight,
)
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

    def index_update(
        self,
        idx: IntArray,
        problem: UpdateProblem,
    ) -> UpdateProblem:
        return IndexProblem(idx, problem)

    def checkerboard_update(
        self,
        problem: UpdateProblem,
    ) -> UpdateProblem:
        return CheckerboardProblem(problem)


#######################
# Custom update specs #
#######################


@Pytree.dataclass(match_args=True)
class StaticResizeProblem(UpdateProblem):
    subproblem: UpdateProblem
    resized_length: Int = Pytree.static()


@Pytree.dataclass(match_args=True)
class IndexProblem(UpdateProblem):
    index: IntArray
    subproblem: UpdateProblem


@Pytree.dataclass(match_args=True)
class CheckerboardProblem(UpdateProblem):
    subproblem: UpdateProblem


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
        constraint: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        (carry, scanned_in) = args

        def _inner_importance(key, constraint, carry, scanned_in):
            tr, w, _retdiff, bwd_problem = self.kernel_gen_fn.update(
                key,
                EmptyTrace(self.kernel_gen_fn),
                GenericProblem(
                    Diff.unknown_change((carry, scanned_in)),
                    ImportanceProblem(constraint),
                ),
            )
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out, w, bwd_problem)

        def _importance(carry, scanned_over):
            key, idx, carried_value = carry
            key = jax.random.fold_in(key, idx)
            submap = constraint.get_submap(idx)
            (carry, score), (tr, scanned_out, w, inner_bwd_problem) = _inner_importance(
                key, submap, carried_value, scanned_over
            )
            bwd_problem = ChoiceMap.idx(idx, inner_bwd_problem)

            return (key, idx + 1, carry), (tr, scanned_out, score, w, bwd_problem)

        (_, _, carried_out), (tr, scanned_out, scores, ws, bwd_problems) = jax.lax.scan(
            _importance,
            (key, 0, carry),
            scanned_in,
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
        )
        return (
            ScanTrace(self, tr, args, (carried_out, scanned_out), jnp.sum(scores)),
            jnp.sum(ws),
            Diff.unknown_change((carried_out, scanned_out)),
            bwd_problems,
        )

    def _get_subproblem(
        self,
        problem: UpdateProblem,
        idx: IntArray,
    ) -> UpdateProblem:
        match problem:
            case ChoiceMap():
                return problem(idx)

            case Selection():
                subproblem = problem(idx)
                return subproblem

            case _:
                raise Exception(f"Not implemented subproblem: {problem}")

    @typecheck
    def update_generic(
        self,
        key: PRNGKey,
        trace: Trace,
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[ScanTrace, Weight, Retdiff, UpdateProblem]:
        carry_diff, *scanned_in_diff = Diff.tree_diff_unknown_change(
            Diff.tree_primal(argdiffs)
        )

        def _inner_update(key, subtrace, subproblem, carry, scanned_in):
            (
                new_subtrace,
                w,
                kernel_retdiff,
                bwd_problem,
            ) = self.kernel_gen_fn.update(
                key,
                subtrace,
                GenericProblem(
                    (carry, scanned_in),
                    subproblem,
                ),
            )
            (carry_retdiff, scanned_out_retdiff) = kernel_retdiff
            score = new_subtrace.get_score()
            return (carry_retdiff, score), (
                new_subtrace,
                scanned_out_retdiff,
                w,
                bwd_problem,
            )

        def _update(carry, scanned_over):
            key, idx, carried_value = carry
            (subtrace, *scanned_in) = scanned_over
            key = jax.random.fold_in(key, idx)
            subproblem = self._get_subproblem(problem, idx)
            (
                (carry, score),
                (new_subtrace, scanned_out, w, inner_bwd_problem),
            ) = _inner_update(key, subtrace, subproblem, carried_value, scanned_in)
            bwd_problem = ChoiceMap.idx(idx, inner_bwd_problem)

            return (key, idx + 1, carry), (
                new_subtrace,
                scanned_out,
                score,
                w,
                bwd_problem,
            )

        (
            (_, _, carried_out_diff),
            (new_subtraces, scanned_out_diff, scores, ws, bwd_problems),
        ) = jax.lax.scan(
            _update,
            (key, 0, carry_diff),
            (trace.inner, *scanned_in_diff),
            length=self.length,
            reverse=self.reverse,
            unroll=self.unroll,
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
            bwd_problems,
        )

    def update_index(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        index: IntArray,
        update_problem: UpdateProblem,
    ):
        starting_subslice = jtu.tree_map(lambda v: v[index], trace.inner)
        affected_subslice = jtu.tree_map(lambda v: v[index + 1], trace.inner)
        starting_argdiffs = Diff.no_change(starting_subslice.get_args())
        (
            updated_start,
            start_w,
            starting_retdiff,
            bwd_problem,
        ) = self.kernel_gen_fn.update(
            key,
            starting_subslice,
            update_problem,
            starting_argdiffs,
        )
        updated_end, end_w, ending_retdiff, _ = self.kernel_gen_fn.update(
            key,
            affected_subslice,
            EmptyProblem(),
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
            IndexProblem(index, bwd_problem),
        )

    @typecheck
    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        assert isinstance(trace, EmptyTrace | ScanTrace)
        match update_problem:
            case ImportanceProblem(constraint) if isinstance(constraint, ChoiceMap):
                return self.update_importance(
                    key, constraint, Diff.tree_primal(argdiffs)
                )
            case IndexProblem(index, subproblem):
                if Diff.static_check_no_change(argdiffs):
                    return self.update_index(key, trace, index, subproblem)
                else:
                    return self.update_generic(
                        key, trace, ChoiceMap.idx(index, subproblem), argdiffs
                    )
            case _:
                return self.update_generic(key, trace, update_problem, argdiffs)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case GenericProblem(argdiffs, subproblem):
                return self.update_change_target(key, trace, subproblem, argdiffs)

            case _:
                return self.update(
                    key,
                    trace,
                    GenericProblem(
                        Diff.no_change(trace.get_args()),
                        update_problem,
                    ),
                )

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


##############
# Decorators #
##############


@typecheck
def scan(
    *, n: Optional[Int] = None, reverse: bool = False, unroll: int | bool = 1
) -> Callable[[GenerativeFunction], ScanCombinator]:
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
