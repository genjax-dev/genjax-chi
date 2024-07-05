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
    List,
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
        )(jnp.arange(self.scan_gen_fn.max_length), self.inner)

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
    """> `ScanCombinator` accepts a kernel_gen_fn generative function, as well as a static
    maximum unroll length, and provides a scan-like pattern of generative computation.

    !!! info "kernel_gen_fn generative functions"
        A kernel_gen_fn generative function is one which accepts and returns the same signature of arguments. Under the hood, `ScanCombinator` is implemented using `jax.lax.scan` - which has the same requirements.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="gen-fn"
        import jax
        import genjax


        # A kernel_gen_fn generative function.
        @genjax.gen
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x


        # You can apply the Scan combinator directly like this:
        scan_gen_fned_random_walk = random_walk.scan(n=1000)


        # You can also use the decorator when declaring the function:
        @genjax.scan(n=1000)
        @genjax.gen
        def random_walk(prev, xs):
            x = genjax.normal(prev, 1.0) @ "x"
            return x, None


        init = 0.5
        key = jax.random.PRNGKey(314159)
        tr = jax.jit(random_walk.simulate)(key, (init, None))

        print(tr.render_html())
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
            length=self.max_length,
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
            length=self.max_length,
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


#############
# Decorator #
#############


@typecheck
def scan(*, n: Int) -> Callable[[GenerativeFunction], ScanCombinator]:
    def decorator(f):
        return ScanCombinator(f, n)

    return decorator
