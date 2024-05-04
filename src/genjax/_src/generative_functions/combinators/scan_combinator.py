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

from genjax._src.core.generative import (
    ChangeTargetUpdateSpec,
    ChoiceMap,
    Constraint,
    GenerativeFunction,
    RemoveSelectionUpdateSpec,
    Retdiff,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
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


@Pytree.dataclass
class ScanTrace(Trace):
    scan_gen_fn: "ScanCombinator"
    inner: Trace
    retval: Any
    score: FloatArray

    def get_sample(self):
        return jax.vmap(
            lambda idx, subtrace: ChoiceMap.a(idx, subtrace.get_sample()),
        )(jnp.arange(self.scan_gen_fn.max_length), self.inner)

    def get_gen_fn(self):
        return self.scan_gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score


#######################
# Custom update specs #
#######################


@Pytree.dataclass
class StaticResizeUpdateSpec(UpdateSpec):
    subspec: UpdateSpec
    resized_length: Int = Pytree.static()


###################
# Scan combinator #
###################


@Pytree.dataclass
class ScanCombinator(GenerativeFunction):
    """> `ScanCombinator` accepts a kernel generative function, as well as a static
    maximum unroll length, and provides a scan-like pattern of generative computation.

    !!! info "Kernel generative functions"
        A kernel generative function is one which accepts and returns the same signature of arguments. Under the hood, `ScanCombinator` is implemented using `jax.lax.scan` - which has the same requirements.

    Examples:
        ```python exec="yes" source="tabbed-left"
        import jax
        import genjax
        console = genjax.console()

        # A kernel generative function.
        @genjax.static_gen_fn
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x

        # You can apply the Scan combinator direclty like this:

        scan_gen_fned_random_walk = genjax.Scan(max_length=1000)(random_walk)

        # But the recommended way to do this is to use `Scan` as a decorator
        # when declaring the function:

        @genjax.scan_combinator(max_length=1000)
        @genjax.static_gen_fn
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x

        init = 0.5
        key = jax.random.PRNGKey(314159)
        tr = jax.jit(genjax.simulate(ramdom_walk)(key, (999, init))

        print(console.render(tr))
        ```
    """

    kernel: GenerativeFunction
    max_length: Int = Pytree.static()

    # To get the type of return value, just invoke
    # the scanned over source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        (carry, scanned_in) = args

        def _inner(carry, scanned_in):
            v, scanned_out = self.kernel.__abstract_call__(carry, scanned_in)
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
            tr = self.gen_fn.simulate(key, (carry, scanned_in))
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

        return ScanTrace(self, tr, (carried_out, scanned_out), jnp.sum(scores))

    def importance_choice_map(
        self,
        key: PRNGKey,
        constraint: ChoiceMap,
        args: Tuple,
    ) -> Tuple[ScanTrace, FloatArray, UpdateSpec]:
        (carry, scanned_in) = args

        def _inner_importance(key, constraint, carry, scanned_in):
            tr, w, bwd_spec = self.kernel.importance(
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
            ScanTrace(self, tr, (carried_out, scanned_out), jnp.sum(scores)),
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
        argdiffs: Tuple,
    ) -> Tuple[ScanTrace, Weight, Retdiff, UpdateSpec]:
        carry_diff, *scanned_in_diff = Diff.tree_diff_unknown_change(argdiffs)

        def _inner_update(key, subtrace, subspec, carry, scanned_in):
            (
                new_subtrace,
                w,
                (carry_retdiff, scanned_out_retdiff),
                bwd_spec,
            ) = self.gen_fn.update(key, subtrace, subspec, (carry, scanned_in))
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
            ScanTrace(self, new_subtraces, (carried_out, scanned_out), jnp.sum(scores)),
            jnp.sum(ws),
            (carried_out_diff, scanned_out_diff),
            bwd_specs,
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
            case _:
                return self.update_generic(key, trace, update_spec, argdiffs)

    def assess(
        self,
        constraint: Constraint,
    ) -> Tuple[FloatArray, Any]:
        raise NotImplementedError


#############
# Decorator #
#############


@typecheck
def scan_combinator(
    gen_fn_closure: Optional[GenerativeFunction] = None,
    /,
    *,
    max_length: Int,
):
    def decorator(f):
        return ScanCombinator(f, max_length)

    if gen_fn_closure:
        return decorator(gen_fn_closure)
    else:
        return decorator
