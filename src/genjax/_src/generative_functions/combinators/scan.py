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
    ChoiceMapConstraint,
    Constraint,
    EditRequest,
    GenerativeFunction,
    IncrementalGenericRequest,
    Projection,
    Retdiff,
    Score,
    Selection,
    Trace,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    Int,
    IntArray,
    PRNGKey,
    TypeVar,
)

Carry = TypeVar("Carry")
Y = TypeVar("Y")


@Pytree.dataclass
class ScanTrace(Generic[Carry, Y], Trace[tuple[Carry, Y]]):
    scan_gen_fn: "ScanCombinator[Carry, Y]"
    inner: Trace[tuple[Carry, Y]]
    args: tuple[Any, ...]
    retval: tuple[Carry, Y]
    score: FloatArray
    chm: ChoiceMap
    scan_length: int = Pytree.static()

    @staticmethod
    def build(
        scan_gen_fn: "ScanCombinator[Carry, Y]",
        inner: Trace[tuple[Carry, Y]],
        args: tuple[Any, ...],
        retval: tuple[Carry, Y],
        score: FloatArray,
        scan_length: int,
    ) -> "ScanTrace[Carry, Y]":
        chm = jax.vmap(
            lambda idx, subtrace: ChoiceMap.entry(subtrace.get_choices(), idx),
        )(jnp.arange(scan_length), inner)

        return ScanTrace(scan_gen_fn, inner, args, retval, score, chm, scan_length)

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> tuple[Carry, Y]:
        return self.retval

    def get_choices(self) -> ChoiceMap:
        return self.chm

    def get_sample(self) -> ChoiceMap:
        return self.get_choices()

    def get_gen_fn(self):
        return self.scan_gen_fn

    def get_score(self):
        return self.score


###################
# Scan combinator #
###################


@Pytree.dataclass
class ScanCombinator(Generic[Carry, Y], GenerativeFunction[tuple[Carry, Y]]):
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

    kernel_gen_fn: GenerativeFunction[tuple[Carry, Y]]

    # Only required for `None` carry inputs
    length: Int | None = Pytree.static()

    # To get the type of return value, just invoke
    # the scanned over source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> tuple[Carry, Y]:
        (carry, scanned_in) = args

        def _inner(carry: Carry, scanned_in: Any):
            v, scanned_out = self.kernel_gen_fn.__abstract_call__(carry, scanned_in)
            return v, scanned_out

        v, scanned_out = jax.lax.scan(_inner, carry, scanned_in, length=self.length)

        return v, scanned_out

    @staticmethod
    def _static_scan_length(xs: Any, length: int | None) -> int:
        # We start by triggering a scan to force all JAX validations to run.
        jax.lax.scan(lambda c, x: (c, None), None, xs, length=length)
        return length or jtu.tree_leaves(xs)[0].shape[0]

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> ScanTrace[Carry, Y]:
        carry, scanned_in = args

        def _inner_simulate(
            key: PRNGKey, carry: Carry, scanned_in: Any
        ) -> tuple[tuple[Carry, Score], tuple[Trace[tuple[Carry, Y]], Y]]:
            tr = self.kernel_gen_fn.simulate(key, (carry, scanned_in))
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out)

        def _inner(
            carry: tuple[PRNGKey, IntArray, Carry], scanned_over: Any
        ) -> tuple[
            tuple[PRNGKey, IntArray, Carry], tuple[Trace[tuple[Carry, Y]], Y, Score]
        ]:
            key, count, carried_value = carry
            key = jax.random.fold_in(key, count)
            (carried_out, score), (tr, scanned_out) = _inner_simulate(
                key, carried_value, scanned_over
            )

            return (key, count + 1, carried_out), (tr, scanned_out, score)

        (_, _, carried_out), (tr, scanned_out, scores) = jax.lax.scan(
            _inner,
            (key, jnp.asarray(0), carry),
            scanned_in,
            length=self.length,
        )

        return ScanTrace.build(
            self,
            tr,
            args,
            (carried_out, scanned_out),
            jnp.sum(scores),
            self._static_scan_length(scanned_in, self.length),
        )

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[ScanTrace[Carry, Y], Weight]:
        assert isinstance(constraint, ChoiceMapConstraint)
        (carry, scanned_in) = args

        def _inner_generate(
            key: PRNGKey,
            constraint: ChoiceMapConstraint,
            carry: Carry,
            scanned_in: Any,
        ) -> tuple[tuple[Carry, Score], tuple[Trace[tuple[Carry, Y]], Y, Weight]]:
            tr, w = self.kernel_gen_fn.generate(
                key,
                constraint,
                (carry, scanned_in),
            )
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out, w)

        def _generate(
            carry: tuple[PRNGKey, IntArray, Carry],
            scanned_over: Any,
        ) -> tuple[
            tuple[PRNGKey, IntArray, Carry],
            tuple[Trace[tuple[Carry, Y]], Y, Score, Weight],
        ]:
            key, idx, carried_value = carry
            key = jax.random.fold_in(key, idx)
            submap = constraint.get_submap(idx)
            assert isinstance(submap, ChoiceMapConstraint)
            (carried_out, score), (tr, scanned_out, w) = _inner_generate(
                key, submap, carried_value, scanned_over
            )

            return (key, idx + 1, carried_out), (tr, scanned_out, score, w)

        (_, _, carried_out), (tr, scanned_out, scores, ws) = jax.lax.scan(
            _generate,
            (key, jnp.asarray(0), carry),
            scanned_in,
            length=self.length,
        )
        return (
            ScanTrace[Carry, Y].build(
                self,
                tr,
                args,
                (carried_out, scanned_out),
                jnp.sum(scores),
                self._static_scan_length(scanned_in, self.length),
            ),
            jnp.sum(ws),
        )

    def project(
        self,
        key: PRNGKey,
        trace: Trace[tuple[Carry, Y]],
        projection: Projection[Any],
    ) -> Weight:
        assert isinstance(projection, Selection)
        assert isinstance(trace, ScanTrace)

        def _inner_project(
            key: PRNGKey,
            subtrace: Trace[Any],
            projection: Selection,
        ) -> Weight:
            w = subtrace.project(
                key,
                projection,
            )
            return w

        def _project(
            carry: tuple[PRNGKey, IntArray],
            subtrace: Trace[Any],
        ) -> tuple[tuple[PRNGKey, IntArray], Weight]:
            key, idx = carry
            key = jax.random.fold_in(key, idx)
            subprojection = projection(idx)
            assert isinstance(subprojection, Selection)
            w = _inner_project(key, subtrace, subprojection)

            return (key, idx + 1), w

        (_, _), ws = jax.lax.scan(
            _project,
            (key, jnp.asarray(0)),
            trace.inner,
            length=self.length,
        )
        return jnp.sum(ws)

    def edit_generic(
        self,
        key: PRNGKey,
        trace: ScanTrace[Carry, Y],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> tuple[ScanTrace[Carry, Y], Weight, Retdiff[tuple[Carry, Y]], EditRequest]:
        assert isinstance(constraint, ChoiceMapConstraint)
        diffs = Diff.tree_diff_unknown_change(Diff.tree_primal(argdiffs))
        carry_diff: Carry = diffs[0]
        scanned_in_diff: Any = diffs[1:]

        def _inner_edit(
            key: PRNGKey,
            subtrace: Trace[tuple[Carry, Y]],
            subconstraint: Constraint,
            carry: Carry,
            scanned_in: Any,
        ) -> tuple[
            tuple[Carry, Score],
            tuple[Trace[tuple[Carry, Y]], Retdiff[Y], Weight, EditRequest],
        ]:
            (
                new_subtrace,
                w,
                kernel_retdiff,
                bwd_request,
            ) = self.kernel_gen_fn.edit(
                key,
                subtrace,
                IncrementalGenericRequest(subconstraint),
                (carry, scanned_in),
            )
            (carry_retdiff, scanned_out_retdiff) = Diff.tree_diff_unknown_change(
                kernel_retdiff
            )
            score = new_subtrace.get_score()
            return (carry_retdiff, score), (
                new_subtrace,
                scanned_out_retdiff,
                w,
                bwd_request,
            )

        def _edit(
            carry: tuple[PRNGKey, IntArray, Carry],
            scanned_over: tuple[Trace[tuple[Carry, Y]], Any],
        ) -> tuple[
            tuple[PRNGKey, IntArray, Carry],
            tuple[
                Trace[tuple[Carry, Y]], Retdiff[Y], Score, Weight, ChoiceMapConstraint
            ],
        ]:
            key, idx, carried_value = carry
            subtrace, scanned_in = scanned_over
            key = jax.random.fold_in(key, idx)
            subconstraint = constraint(idx)
            assert isinstance(subconstraint, ChoiceMapConstraint)
            (
                (carried_out, score),
                (new_subtrace, scanned_out, w, inner_bwd_request),
            ) = _inner_edit(key, subtrace, subconstraint, carried_value, scanned_in)
            assert isinstance(inner_bwd_request, IncrementalGenericRequest)
            bwd_constraint = inner_bwd_request.constraint
            bwd_constraint = ChoiceMapConstraint(ChoiceMap.entry(bwd_constraint, idx))

            return (key, idx + 1, carried_out), (
                new_subtrace,
                scanned_out,
                score,
                w,
                bwd_constraint,
            )

        (
            (_, _, carried_out_diff),
            (new_subtraces, scanned_out_diff, scores, ws, bwd_constraints),
        ) = jax.lax.scan(
            _edit,
            (key, jnp.asarray(0), carry_diff),
            (trace.inner, *scanned_in_diff),
            length=self.length,
        )
        carried_out, scanned_out = Diff.tree_primal((
            carried_out_diff,
            scanned_out_diff,
        ))
        return (
            ScanTrace.build(
                self,
                new_subtraces,
                Diff.tree_primal(argdiffs),
                (carried_out, scanned_out),
                jnp.sum(scores),
                trace.scan_length,
            ),
            jnp.sum(ws),
            (carried_out_diff, scanned_out_diff),
            IncrementalGenericRequest(bwd_constraints),
        )

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[tuple[Carry, Y]],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[ScanTrace[Carry, Y], Weight, Retdiff[tuple[Carry, Y]], EditRequest]:
        assert isinstance(edit_request, IncrementalGenericRequest)
        assert isinstance(trace, ScanTrace)
        return self.edit_generic(
            key,
            trace,
            edit_request.constraint,
            argdiffs,
        )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, Any]:
        (carry, scanned_in) = args

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
        )
        return (
            jnp.sum(scores),
            (carried_out, scanned_out),
        )


##############
# Decorators #
##############


def scan(
    *, n: Int | None = None
) -> Callable[
    [GenerativeFunction[tuple[Carry, Y]]], GenerativeFunction[tuple[Carry, Y]]
]:
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

    def decorator(f: GenerativeFunction[tuple[Carry, Y]]):
        return ScanCombinator[Carry, Y](f, length=n)

    return decorator


def prepend_initial_acc(args: tuple[Carry, Any], ret: tuple[Carry, Carry]) -> Carry:
    """Prepends the initial accumulator value to the array of accumulated
    values.

    This function is used in the context of scan operations to include the initial
    accumulator state in the output, effectively providing a complete history of
    the accumulator's values throughout the scan.

    Args:
        args: A tuple containing the initial arguments to the scan operation. The first element is expected to be the initial accumulator value.
        ret: A tuple containing the final accumulator value and an array of intermediate accumulator values from the scan operation.

    Returns:
        A tree structure where each leaf is an array with the initial accumulator value prepended to the corresponding array of intermediate values.

    Note:
        This function uses JAX's tree mapping to handle nested structures in the accumulator, allowing it to work with complex accumulator types.
    """
    init_acc = args[0]
    xs = ret[1]

    def cat(init, arr):
        return jnp.concatenate([jnp.array(init)[jnp.newaxis], arr])

    return jax.tree.map(cat, init_acc, xs)


def accumulate() -> Callable[[GenerativeFunction[Carry]], GenerativeFunction[Carry]]:
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

    def decorator(f: GenerativeFunction[Carry]) -> GenerativeFunction[Carry]:
        return (
            f.map(lambda ret: (ret, ret))
            .scan()
            .dimap(pre=lambda *args: args, post=prepend_initial_acc)
        )

    return decorator


def reduce() -> Callable[[GenerativeFunction[Carry]], GenerativeFunction[Carry]]:
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

    def decorator(f: GenerativeFunction[Carry]) -> GenerativeFunction[Carry]:
        def pre(ret: Carry):
            return ret, None

        def post(ret: tuple[Carry, None]):
            return ret[0]

        return f.map(pre).scan().map(post)

    return decorator


def iterate(*, n: Int) -> Callable[[GenerativeFunction[Y]], GenerativeFunction[Y]]:
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

    def decorator(f: GenerativeFunction[Y]) -> GenerativeFunction[Y]:
        # strip off the JAX-supplied `None` on the way in, accumulate `ret` on the way out.
        return (
            f.dimap(pre=lambda *args: args[:-1], post=lambda _, ret: (ret, ret))
            .scan(n=n)
            .dimap(pre=lambda *args: (*args, None), post=prepend_initial_acc)
        )

    return decorator


def iterate_final(
    *, n: Int
) -> Callable[[GenerativeFunction[Y]], GenerativeFunction[Y]]:
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

    def decorator(f: GenerativeFunction[Y]) -> GenerativeFunction[Y]:
        # strip off the JAX-supplied `None` on the way in, no accumulation on the way out.
        def pre_post(_, ret: Y):
            return ret, None

        def post_post(_, ret: tuple[Y, None]):
            return ret[0]

        return (
            f.dimap(pre=lambda *args: args[:-1], post=pre_post)
            .scan(n=n)
            .dimap(pre=lambda *args: (*args, None), post=post_post)
        )

    return decorator
