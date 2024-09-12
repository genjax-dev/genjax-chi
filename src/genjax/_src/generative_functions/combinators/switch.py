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
    Constraint,
    EditRequest,
    GenerativeFunction,
    IncrementalGenericRequest,
    Projection,
    Retdiff,
    Sample,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.generative.functional_types import staged_choose
from genjax._src.core.interpreters.incremental import Diff, NoChange, UnknownChange
from genjax._src.core.interpreters.staging import get_data_shape
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    FloatArray,
    Generic,
    Int,
    IntArray,
    PRNGKey,
    Sequence,
    TypeVar,
)

R = TypeVar("R")

#######################
# Switch sample types #
#######################


@Pytree.dataclass
class HeterogeneousSwitchSample(Sample):
    index: IntArray
    subtraces: Sequence[ChoiceMap]


################
# Switch trace #
################


@Pytree.dataclass
class SwitchTrace(Generic[R], Trace[R]):
    gen_fn: "SwitchCombinator[R]"
    args: tuple[Any, ...]
    subtraces: list[Trace[R]]
    retval: R
    score: FloatArray

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_choices(self) -> ChoiceMap:
        (idx, *_) = self.get_args()
        sub_chms = list(map(lambda v: v.get_choices(), self.subtraces))
        chm = ChoiceMap.empty()
        for _idx, _chm in enumerate(sub_chms):
            assert isinstance(_chm, ChoiceMap)
            masked_submap = _chm.mask(jnp.all(_idx == idx))
            chm = chm ^ masked_submap
        return chm

    def get_sample(self) -> ChoiceMap:
        subsamples = list(map(lambda v: v.get_sample(), self.subtraces))
        (idx, *_) = self.get_args()
        chm = ChoiceMap.empty()
        for _idx, _chm in enumerate(subsamples):
            assert isinstance(_chm, ChoiceMap)
            masked_submap = _chm.mask(jnp.all(_idx == idx))
            chm = chm ^ masked_submap
        return chm

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score


#####################
# Switch combinator #
#####################


@Pytree.dataclass
class SwitchCombinator(Generic[R], GenerativeFunction[R]):
    """
    `SwitchCombinator` accepts `n` generative functions as input and returns a new [`genjax.GenerativeFunction`][] that accepts `n+1` arguments:

    - an index in the range `[0, n-1]`
    - a tuple of arguments for each of the input generative functions

    and executes the generative function at the supplied index with its provided arguments.

    If `index` is out of bounds, `index` is clamped to within bounds.

    !!! info "Existence uncertainty"

        This pattern allows `GenJAX` to express existence uncertainty over random choices -- as different generative function branches need not share addresses.

    Attributes:
        branches: generative functions that the `SwitchCombinator` will select from based on the supplied index.

    Examples:
        Create a `SwitchCombinator` via the [`genjax.switch`][] method:
        ```python exec="yes" html="true" source="material-block" session="switch"
        import jax, genjax


        @genjax.gen
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"


        @genjax.gen
        def branch_2():
            x = genjax.bernoulli(0.3) @ "x2"


        switch = genjax.switch(branch_1, branch_2)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)

        # Select `branch_2` by providing 1:
        tr = jitted(key, (1, (), ()))

        print(tr.render_html())
        ```
    """

    branches: tuple[GenerativeFunction[R], ...]

    def __abstract_call__(self, *args) -> R:
        idx, args = args[0], args[1:]
        retvals: list[R] = []
        for _idx in range(len(self.branches)):
            branch_gen_fn = self.branches[_idx]
            branch_args = args[_idx]
            retval = branch_gen_fn.__abstract_call__(*branch_args)
            retvals.append(retval)
        return staged_choose(idx, retvals)

    def static_check_num_arguments_equals_num_branches(self, args):
        assert len(args) == len(self.branches)

    def _empty_simulate_defs(
        self,
        args: tuple[Any, ...],
    ):
        trace_defs = []
        trace_leaves = []
        retval_defs = []
        retval_leaves = []
        for static_idx in range(len(self.branches)):
            key = jax.random.PRNGKey(0)
            branch_gen_fn = self.branches[static_idx]
            branch_args = args[static_idx]
            trace_shape = get_data_shape(branch_gen_fn.simulate)(key, branch_args)
            empty_trace = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), trace_shape
            )
            retval_leaf, retval_def = jtu.tree_flatten(empty_trace.get_retval())
            trace_leaf, trace_def = jtu.tree_flatten(empty_trace)
            trace_defs.append(trace_def)
            trace_leaves.append(trace_leaf)
            retval_defs.append(retval_def)
            retval_leaves.append(retval_leaf)
        return (trace_leaves, trace_defs), (retval_leaves, retval_defs)

    def _simulate(self, trace_leaves, retval_leaves, key, static_idx, args):
        branch_gen_fn = self.branches[static_idx]
        args = args[static_idx]
        tr = branch_gen_fn.simulate(key, args)
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retval_leaves[static_idx] = jtu.tree_leaves(tr.get_retval())
        score = tr.get_score()
        return (trace_leaves, retval_leaves), score

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> SwitchTrace[R]:
        idx: ArrayLike = args[0]
        branch_args = args[1:]

        self.static_check_num_arguments_equals_num_branches(branch_args)

        def _inner(idx: int):
            return lambda trace_leaves, retval_leaves, key, args: self._simulate(
                trace_leaves, retval_leaves, key, idx, args
            )

        branch_functions = list(map(_inner, range(len(self.branches))))
        (
            (trace_leaves, trace_defs),
            (retval_leaves, retval_defs),
        ) = self._empty_simulate_defs(branch_args)
        (trace_leaves, retval_leaves), score = jax.lax.switch(
            idx, branch_functions, trace_leaves, retval_leaves, key, branch_args
        )
        subtraces = list(
            map(
                lambda x: jtu.tree_unflatten(trace_defs[x], trace_leaves[x]),
                range(len(trace_leaves)),
            )
        )
        retvals: list[R] = list(
            map(
                lambda x: jtu.tree_unflatten(retval_defs[x], retval_leaves[x]),
                range(len(retval_leaves)),
            )
        )
        retval: R = staged_choose(idx, retvals)
        return SwitchTrace(self, args, subtraces, retval, score)

    def _empty_assess_defs(self, sample: Sample, args: tuple[Any, ...]):
        retval_defs = []
        retval_leaves = []
        for static_idx in range(len(self.branches)):
            branch_gen_fn = self.branches[static_idx]
            branch_args = args[static_idx]
            _, retval_shape = get_data_shape(branch_gen_fn.assess)(sample, branch_args)
            empty_retval = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), retval_shape
            )
            retval_leaf, retval_def = jtu.tree_flatten(empty_retval)
            retval_defs.append(retval_def)
            retval_leaves.append(retval_leaf)
        return (retval_leaves, retval_defs)

    def _assess(self, static_idx, sample, args):
        branch_gen_fn = self.branches[static_idx]
        branch_args = args[static_idx]
        score, retval = branch_gen_fn.assess(sample, branch_args)
        (retval_leaves, _) = self._empty_assess_defs(sample, args)
        retval_leaves[static_idx] = jtu.tree_leaves(retval)
        return retval_leaves, score

    def assess(
        self,
        sample: Sample,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        assert isinstance(sample, ChoiceMap)
        idx, branch_args = args[0], args[1:]
        self.static_check_num_arguments_equals_num_branches(branch_args)

        def _inner(static_idx: int):
            return lambda sample, args: self._assess(static_idx, sample, args)

        branch_functions = list(map(_inner, range(len(self.branches))))

        retval_leaves, score = jax.lax.switch(
            idx, branch_functions, sample, branch_args
        )
        (_, retval_defs) = self._empty_assess_defs(sample, branch_args)
        retvals = list(
            map(
                lambda x: jtu.tree_unflatten(retval_defs[x], retval_leaves[x]),
                range(len(retval_leaves)),
            )
        )
        retval: R = staged_choose(idx, retvals)
        return score, retval

    def _empty_generate_defs(
        self,
        constraint: Constraint,
        args: tuple[Any, ...],
    ):
        trace_defs = []
        trace_leaves = []
        retval_defs = []
        retval_leaves = []
        for static_idx in range(len(self.branches)):
            branch_gen_fn = self.branches[static_idx]
            branch_args = args[static_idx]
            key = jax.random.PRNGKey(0)
            trace_shape, _ = get_data_shape(branch_gen_fn.generate)(
                key,
                constraint,
                branch_args,
            )
            empty_trace = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), trace_shape
            )
            trace_leaf, trace_def = jtu.tree_flatten(empty_trace)
            retval_leaf, retval_def = jtu.tree_flatten(empty_trace.get_retval())
            retval_defs.append(retval_def)
            retval_leaves.append(retval_leaf)
            trace_defs.append(trace_def)
            trace_leaves.append(trace_leaf)
        return (
            (trace_leaves, trace_defs),
            (retval_leaves, retval_defs),
        )

    def _generate(
        self,
        trace_leaves,
        retval_leaves,
        key,
        static_idx: int,
        constraint,
        args,
    ):
        branch_gen_fn = self.branches[static_idx]
        branch_args = args[static_idx]
        tr, w = branch_gen_fn.generate(
            key,
            constraint,
            branch_args,
        )
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retval_leaves[static_idx] = jtu.tree_leaves(tr.get_retval())
        score = tr.get_score()
        return (trace_leaves, retval_leaves), (score, w)

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[SwitchTrace[R], Weight]:
        (idx, *branch_args) = args
        (_, *branch_args) = args
        branch_args = tuple(branch_args)
        self.static_check_num_arguments_equals_num_branches(branch_args)

        def _inner(static_idx: int):
            return (
                lambda trace_leaves,
                retval_leaves,
                key,
                problem,
                branch_args: self._generate(
                    trace_leaves,
                    retval_leaves,
                    key,
                    static_idx,
                    problem,
                    branch_args,
                )
            )

        branch_functions = list(map(_inner, range(len(self.branches))))
        (
            (trace_leaves, trace_defs),
            (retval_leaves, retval_defs),
        ) = self._empty_generate_defs(constraint, branch_args)

        (trace_leaves, retval_leaves), (score, w) = jax.lax.switch(
            idx,
            branch_functions,
            trace_leaves,
            retval_leaves,
            key,
            constraint,
            branch_args,
        )
        subtraces = list(
            map(
                lambda x: jtu.tree_unflatten(trace_defs[x], trace_leaves[x]),
                range(len(trace_leaves)),
            )
        )
        retvals = list(
            map(
                lambda x: jtu.tree_unflatten(retval_defs[x], retval_leaves[x]),
                range(len(retval_leaves)),
            )
        )
        retval = staged_choose(idx, retvals)
        return (SwitchTrace(self, args, subtraces, retval, score), w)

    def project(
        self,
        key: PRNGKey,
        trace: Trace[R],
        projection: Projection[Any],
    ) -> Weight:
        raise NotImplementedError

    def _empty_edit_defs(
        self,
        trace: SwitchTrace[R],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ):
        trace_defs = []
        trace_leaves = []
        bwd_request_defs = []
        bwd_request_leaves = []
        retdiff_defs = []
        retdiff_leaves = []
        for static_idx in range(len(self.branches)):
            subtrace = trace.subtraces[static_idx]
            gen_fn = self.branches[static_idx]
            branch_argdiffs = argdiffs[static_idx]
            key = jax.random.PRNGKey(0)
            trace_shape, _, retdiff_shape, bwd_request_shape = get_data_shape(
                gen_fn.edit
            )(key, subtrace, IncrementalGenericRequest(constraint), branch_argdiffs)
            empty_trace = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), trace_shape
            )
            empty_retdiff = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), retdiff_shape
            )
            empty_problem = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), bwd_request_shape
            )
            trace_leaf, trace_def = jtu.tree_flatten(empty_trace)
            bwd_request_leaf, bwd_request_def = jtu.tree_flatten(empty_problem)
            retdiff_leaf, retdiff_def = jtu.tree_flatten(empty_retdiff)
            trace_defs.append(trace_def)
            trace_leaves.append(trace_leaf)
            bwd_request_defs.append(bwd_request_def)
            bwd_request_leaves.append(bwd_request_leaf)
            retdiff_defs.append(retdiff_def)
            retdiff_leaves.append(retdiff_leaf)
        return (
            (trace_leaves, trace_defs),
            (retdiff_leaves, retdiff_defs),
            (bwd_request_leaves, bwd_request_defs),
        )

    def _specialized_edit_idx_no_change(
        self,
        key: PRNGKey,
        static_idx: Int,
        trace: SwitchTrace[R],
        constraint: Constraint,
        idx: IntArray,
        argdiffs: Argdiffs,
    ):
        subtrace = trace.subtraces[static_idx]
        gen_fn = self.branches[static_idx]
        branch_argdiffs = argdiffs[static_idx]
        tr, w, rd, bwd_request = gen_fn.edit(
            key,
            subtrace,
            IncrementalGenericRequest(constraint),
            branch_argdiffs,
        )
        (
            (trace_leaves, _),
            (retdiff_leaves, _),
            (bwd_request_leaves, _),
        ) = self._empty_edit_defs(trace, constraint, argdiffs)
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retdiff_leaves[static_idx] = jtu.tree_leaves(rd)
        bwd_request_leaves[static_idx] = jtu.tree_leaves(bwd_request)
        score = tr.get_score()
        return (trace_leaves, retdiff_leaves, bwd_request_leaves), (score, w)

    def _generic_edit_idx_change(
        self,
        key: PRNGKey,
        static_idx: Int,
        trace: SwitchTrace[R],
        constraint: Constraint,
        idx: IntArray,
        argdiffs: Argdiffs,
    ):
        gen_fn = self.branches[static_idx]
        branch_argdiffs = argdiffs[static_idx]
        check = static_idx == idx
        branch_primals = Diff.tree_primal(branch_argdiffs)
        new_subtrace = gen_fn.simulate(key, branch_primals)
        new_subtrace_def = jtu.tree_structure(new_subtrace)
        _, _, _, bwd_request_shape = get_data_shape(gen_fn.edit)(
            key,
            new_subtrace,
            IncrementalGenericRequest(constraint),
            branch_argdiffs,
        )
        bwd_request_def = jtu.tree_structure(bwd_request_shape)

        def _edit_same_branch(key, subtrace, constraint, branch_argdiffs):
            tr, w, rd, bwd_request = gen_fn.edit(
                key,
                subtrace,
                IncrementalGenericRequest(constraint),
                branch_argdiffs,
            )
            rd = Diff.tree_diff_unknown_change(rd)
            tr_leaves = jtu.tree_leaves(tr)
            problem_leaves = jtu.tree_leaves(bwd_request)
            return tr_leaves, w, rd, problem_leaves

        def _edit_new_branch(key, subtrace, constraint, branch_argdiffs):
            branch_argdiffs = Diff.tree_diff_no_change(branch_argdiffs)
            tr, w, rd, bwd_request = gen_fn.edit(
                key,
                subtrace,
                IncrementalGenericRequest(constraint),
                branch_argdiffs,
            )
            rd = Diff.tree_diff_unknown_change(rd)
            tr_leaves = jtu.tree_leaves(tr)
            problem_leaves = jtu.tree_leaves(bwd_request)
            return tr_leaves, w, rd, problem_leaves

        tr_leaves, w, rd, bwd_request_leaves = jax.lax.cond(
            check,
            _edit_same_branch,
            _edit_new_branch,
            key,
            new_subtrace,
            constraint,
            branch_argdiffs,
        )
        tr = jtu.tree_unflatten(new_subtrace_def, tr_leaves)
        bwd_request = jtu.tree_unflatten(bwd_request_def, bwd_request_leaves)
        (
            (trace_leaves, _),
            (retdiff_leaves, _),
            (bwd_request_leaves, _),
        ) = self._empty_edit_defs(trace, constraint, argdiffs)
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retdiff_leaves[static_idx] = jtu.tree_leaves(rd)
        bwd_request_leaves[static_idx] = jtu.tree_leaves(bwd_request)
        score = tr.get_score()
        return (trace_leaves, retdiff_leaves, bwd_request_leaves), (score, w)

    def edit_generic(
        self,
        key: PRNGKey,
        trace: SwitchTrace[R],
        constraint: Constraint,
        argdiffs: Argdiffs,
    ) -> tuple[SwitchTrace[R], Weight, Retdiff[R], EditRequest]:
        (idx_argdiff, *branch_argdiffs) = argdiffs
        self.static_check_num_arguments_equals_num_branches(branch_argdiffs)

        def edit_dispatch(static_idx: int):
            if Diff.tree_tangent(idx_argdiff) == NoChange:
                return (
                    lambda key,
                    trace,
                    problem,
                    idx,
                    argdiffs: self._specialized_edit_idx_no_change(
                        key, static_idx, trace, problem, idx, argdiffs
                    )
                )
            else:
                return (
                    lambda key,
                    trace,
                    problem,
                    idx,
                    argdiffs: self._generic_edit_idx_change(
                        key, static_idx, trace, problem, idx, argdiffs
                    )
                )

        primals = Diff.tree_primal(argdiffs)
        idx = primals[0]
        branch_functions = list(map(edit_dispatch, range(len(self.branches))))

        (trace_leaves, retdiff_leaves, bwd_request_leaves), (score, w) = jax.lax.switch(
            idx, branch_functions, key, trace, constraint, idx, tuple(branch_argdiffs)
        )
        (
            (_, trace_defs),
            (_, retdiff_defs),
            (_, bwd_request_defs),
        ) = self._empty_edit_defs(trace, constraint, tuple(branch_argdiffs))
        subtraces = list(
            map(
                lambda x: jtu.tree_unflatten(trace_defs[x], trace_leaves[x]),
                range(len(trace_leaves)),
            )
        )
        retdiffs = list(
            map(
                lambda x: jtu.tree_unflatten(retdiff_defs[x], retdiff_leaves[x]),
                range(len(retdiff_leaves)),
            )
        )
        bwd_requests = list(
            map(
                lambda x: jtu.tree_unflatten(
                    bwd_request_defs[x], bwd_request_leaves[x]
                ),
                range(len(bwd_request_leaves)),
            )
        )
        retdiff: R = staged_choose(idx_argdiff.primal, retdiffs)
        retval: R = Diff.tree_primal(retdiff)
        if Diff.tree_tangent(idx_argdiff) == UnknownChange:
            w = w + (score - trace.get_score())

        # TODO: this is totally wrong, fix in future PR.
        bwd_request = IncrementalGenericRequest(
            bwd_requests[0].constraint,
        )

        return (
            SwitchTrace(self, primals, subtraces, retval, score),
            w,
            retdiff,
            bwd_request,
        )

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[SwitchTrace[R], Weight, Retdiff[R], EditRequest]:
        assert isinstance(edit_request, IncrementalGenericRequest)
        assert isinstance(trace, SwitchTrace)
        return self.edit_generic(
            key,
            trace,
            edit_request.constraint,
            argdiffs,
        )


#############
# Decorator #
#############


def switch(
    *gen_fns: GenerativeFunction[R],
) -> SwitchCombinator[R]:
    """
    Given `n` [`genjax.GenerativeFunction`][] inputs, returns a [`genjax.GenerativeFunction`][] that accepts `n+1` arguments:

    - an index in the range $[0, n)$
    - a tuple of arguments for each of the input generative functions (`n` total tuples)

    and executes the generative function at the supplied index with its provided arguments.

    If `index` is out of bounds, `index` is clamped to within bounds.

    Args:
        gen_fns: generative functions that the `SwitchCombinator` will select from.

    Returns:




    Examples:
        Create a `SwitchCombinator` via the [`genjax.switch`][] method:
        ```python exec="yes" html="true" source="material-block" session="switch"
        import jax, genjax


        @genjax.gen
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"


        @genjax.gen
        def branch_2():
            x = genjax.bernoulli(0.3) @ "x2"


        switch = genjax.switch(branch_1, branch_2)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)

        # Select `branch_2` by providing 1:
        tr = jitted(key, (1, (), ()))

        print(tr.render_html())
        ```
    """
    return SwitchCombinator[R](gen_fns)
