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
    Address,
    Argdiffs,
    ChoiceMap,
    Constraint,
    EmptyUpdateSpec,
    GenerativeFunction,
    RemoveSelectionUpdateSpec,
    Retdiff,
    Score,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
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
            lambda idx, subtrace: ChoiceMap.a(idx, subtrace.get_sample()),
        )(jnp.arange(self.scan_gen_fn.max_length), self.inner)

    def get_gen_fn(self):
        return self.scan_gen_fn

    def get_score(self):
        return self.score

    def index_update(self, key: PRNGKey, spec: UpdateSpec):
        return self.update(key, spec)

    def create_update_spec(self, addr: Address, v) -> UpdateSpec:
        (idx, *rest) = addr
        return IndexUpdateSpec(idx, self.inner.create_update_spec(tuple(rest), v))


#######################
# Custom update specs #
#######################


@Pytree.dataclass(match_args=True)
class StaticResizeUpdateSpec(UpdateSpec):
    subspec: UpdateSpec
    resized_length: Int = Pytree.static()


@Pytree.dataclass(match_args=True)
class IndexUpdateSpec(UpdateSpec):
    index: IntArray
    subspec: UpdateSpec


###################
# Scan combinator #
###################


@Pytree.dataclass
class ScanCombinator(GenerativeFunction):
    """> `ScanCombinator` accepts a kernel_gen_fn generative function, as well as a static
    maximum unroll length, and provides a scan-like pattern of generative computation.

    !!! info "kernel_gen_fn generative functions"
        A kernel_gen_fn generative function is one which accepts and returns the same signature of arguments. Under the hood, `ScanCombinator` is implemented using `jax.lax.scan` - which has the same requirements.

    Examples:
        ```python exec="yes" source="tabbed-left"
        import jax
        import genjax
        console = genjax.console()

        # A kernel_gen_fn generative function.
        @genjax.gen
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x

        # You can apply the Scan combinator directly like this:

        scan_gen_fned_random_walk = genjax.Scan(max_length=1000)(random_walk)

        # But the recommended way to do this is to use `Scan` as a decorator
        # when declaring the function:

        @genjax.scan_combinator(max_length=1000)
        @genjax.gen
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x

        init = 0.5
        key = jax.random.PRNGKey(314159)
        tr = jax.jit(genjax.simulate(ramdom_walk)(key, (999, init))

        print(console.render(tr))
        ```
    """

    kernel_gen_fn: GenerativeFunction
    max_length: Int = Pytree.static()

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
            length=self.max_length,
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
            length=self.max_length,
        )

        return ScanTrace(self, tr, args, (carried_out, scanned_out), jnp.sum(scores))

    def importance_choice_map(
        self,
        key: PRNGKey,
        constraint: ChoiceMap,
        args: Tuple,
    ) -> Tuple[ScanTrace, FloatArray, UpdateSpec]:
        (carry, scanned_in) = args

        def _inner_importance(key, constraint, carry, scanned_in):
            tr, w, bwd_spec = self.kernel_gen_fn.importance(
                key, constraint, (carry, scanned_in)
            )
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out, w, bwd_spec)

        def _importance(carry, scanned_over):
            key, idx, carried_value = carry
            key = jax.random.fold_in(key, idx)
            submap = constraint.get_submap(idx)
            (carry, score), (tr, scanned_out, w, inner_bwd_spec) = _inner_importance(
                key, submap, carried_value, scanned_over
            )
            bwd_spec = ChoiceMap.a(idx, inner_bwd_spec)

            return (key, idx + 1, carry), (tr, scanned_out, score, w, bwd_spec)

        (_, _, carried_out), (tr, scanned_out, scores, ws, bwd_specs) = jax.lax.scan(
            _importance,
            (key, 0, carry),
            scanned_in,
            length=self.max_length,
        )
        return (
            ScanTrace(self, tr, args, (carried_out, scanned_out), jnp.sum(scores)),
            jnp.sum(ws),
            bwd_specs,
        )

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[ScanTrace, Weight, UpdateSpec]:
        match constraint:
            case ChoiceMap():
                return self.importance_choice_map(key, constraint, args)

            case _:
                raise NotImplementedError

    def _get_subspec(
        self,
        spec: UpdateSpec,
        idx: IntArray,
    ) -> UpdateSpec:
        match spec:
            case ChoiceMap():
                return spec.get_submap(idx)

            case RemoveSelectionUpdateSpec(selection):
                subselection = selection.step(idx)
                return RemoveSelectionUpdateSpec(subselection)

            case _:
                raise Exception(f"Not implemented subspec: {spec}")

    @typecheck
    def update_generic(
        self,
        key: PRNGKey,
        trace: Trace,
        spec: UpdateSpec,
        argdiffs: Argdiffs,
    ) -> Tuple[ScanTrace, Weight, Retdiff, UpdateSpec]:
        carry_diff, *scanned_in_diff = Diff.tree_diff_unknown_change(
            Diff.tree_primal(argdiffs)
        )

        def _inner_update(key, subtrace, subspec, carry, scanned_in):
            (
                new_subtrace,
                w,
                kernel_retdiff,
                bwd_spec,
            ) = self.kernel_gen_fn.update(key, subtrace, subspec, (carry, scanned_in))
            (carry_retdiff, scanned_out_retdiff) = kernel_retdiff
            score = new_subtrace.get_score()
            return (carry_retdiff, score), (
                new_subtrace,
                scanned_out_retdiff,
                w,
                bwd_spec,
            )

        def _update(carry, scanned_over):
            key, idx, carried_value = carry
            (subtrace, *scanned_in) = scanned_over
            key = jax.random.fold_in(key, idx)
            subspec = self._get_subspec(spec, idx)
            (
                (carry, score),
                (new_subtrace, scanned_out, w, inner_bwd_spec),
            ) = _inner_update(key, subtrace, subspec, carried_value, scanned_in)
            bwd_spec = ChoiceMap.a(idx, inner_bwd_spec)

            return (key, idx + 1, carry), (
                new_subtrace,
                scanned_out,
                score,
                w,
                bwd_spec,
            )

        (
            (_, _, carried_out_diff),
            (new_subtraces, scanned_out_diff, scores, ws, bwd_specs),
        ) = jax.lax.scan(
            _update,
            (key, 0, carry_diff),
            (trace.inner, *scanned_in_diff),
            length=self.max_length,
        )
        carried_out, scanned_out = Diff.tree_primal(
            (carried_out_diff, scanned_out_diff)
        )
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
            bwd_specs,
        )

    def update_index(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        index: IntArray,
        update_spec: UpdateSpec,
    ):
        starting_subslice = jtu.tree_map(lambda v: v[index], trace.inner)
        affected_subslice = jtu.tree_map(lambda v: v[index + 1], trace.inner)
        argdiffs = Diff.no_change(starting_subslice.get_args())
        updated_start, start_w, starting_retdiff, bwd_spec = self.kernel_gen_fn.update(
            key, starting_subslice, update_spec, argdiffs
        )
        updated_end, end_w, ending_retdiff, bwd_spec = self.kernel_gen_fn.update(
            key, affected_subslice, EmptyUpdateSpec(), starting_retdiff
        )

        # Must be true for this type of update to be valid.
        assert Diff.static_check_no_change(ending_retdiff)

        def _mutate_in_place(arr, updated_start, updated_end):
            arr = arr.at[index].set(updated_start)
            arr = arr.at[index + 1].set(updated_end)

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
            ChoiceMap.a(index, bwd_spec),
        )

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: ScanTrace,
        update_spec: UpdateSpec,
        argdiffs: Tuple,
    ) -> Tuple[ScanTrace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case IndexUpdateSpec(index, subspec):
                if Diff.static_check_no_change(argdiffs):
                    return self.update_index(key, trace, index, subspec)
                else:
                    return self.update_generic(
                        key, trace, ChoiceMap.a(index, subspec), argdiffs
                    )
            case _:
                return self.update_generic(key, trace, update_spec, argdiffs)

    @typecheck
    def assess(
        self,
        sample: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Score, Any]:
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
            length=self.max_length,
        )
        return (
            jnp.sum(scores),
            (carried_out, scanned_out),
        )


#############
# Decorator #
#############


@typecheck
def scan_combinator(
    gen_fn: Optional[GenerativeFunction] = None,
    /,
    *,
    max_length: Int,
):
    def decorator(f):
        return ScanCombinator(f, max_length)

    if gen_fn:
        return decorator(gen_fn)
    else:
        return decorator
