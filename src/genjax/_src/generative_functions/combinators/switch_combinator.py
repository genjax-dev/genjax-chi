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
    EmptyTrace,
    GenerativeFunction,
    ImportanceProblem,
    Retdiff,
    Sample,
    Score,
    Sum,
    SumConstraint,
    SumProblem,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff, NoChange, UnknownChange
from genjax._src.core.interpreters.staging import get_data_shape
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    FloatArray,
    Int,
    IntArray,
    List,
    PRNGKey,
    Sequence,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


#######################
# Switch sample types #
#######################


@Pytree.dataclass
class HeterogeneousSwitchSample(Sample):
    index: IntArray
    subtraces: Sequence[Sample]

    def get_constraint(self):
        return SumConstraint(
            self.index,
            list(map(lambda x: x.get_constraint(), self.subtraces)),
        )


################
# Switch trace #
################


@Pytree.dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    subtraces: List[Trace]
    retval: Any
    score: FloatArray

    def get_args(self) -> Tuple:
        return self.args

    def get_sample(self) -> Sample:
        subsamples = list(map(lambda v: v.get_sample(), self.subtraces))
        if all(map(lambda v: isinstance(v, ChoiceMap), subsamples)):
            (idx, *_) = self.get_args()
            chm = ChoiceMap.empty()
            for _idx, _chm in enumerate(subsamples):
                assert isinstance(_chm, ChoiceMap)
                masked_submap = ChoiceMap.maybe(_idx == idx, _chm)
                chm = chm ^ masked_submap
            return chm
        else:
            (idx, *_) = self.args
            return HeterogeneousSwitchSample(
                idx,
                list(
                    map(lambda tr: tr.get_sample(), self.subtraces),
                ),
            )

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
class SwitchCombinator(GenerativeFunction):
    """> `SwitchCombinator` accepts multiple generative functions as input and
    implements `GenerativeFunction` interface semantics that support branching control
    flow patterns, including control flow patterns which branch on other stochastic
    choices.

    !!! info "Existence uncertainty"

        This pattern allows `GenJAX` to express existence uncertainty over random choices -- as different generative function branches need not share addresses.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="gen-fn"
        import jax
        import genjax

        @genjax.gen
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"


        @genjax.gen
        def branch_2():
            x = genjax.bernoulli(0.3) @ "x2"


        ################################################################################
        # Creating a `SwitchCombinator` via the preferred `switch_combinator` function #
        ################################################################################

        switch = genjax.switch_combinator(branch_1, branch_2)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        _ = jitted(key, (0, (), ()))
        tr = jitted(key, (1, (), ()))

        print(tr.render_html())
        ```
    """

    branches: Tuple[GenerativeFunction, ...]

    def __abstract_call__(self, idx, *args):
        retvals = []
        for _idx in range(len(self.branches)):
            branch_gen_fn = self.branches[_idx]
            branch_args = args[_idx]
            retval = branch_gen_fn.__abstract_call__(*branch_args)
            retvals.append(retval)
        return Sum.maybe(idx, retvals)

    def static_check_num_arguments_equals_num_branches(self, args):
        assert len(args) == len(self.branches)

    @typecheck
    def _empty_simulate_defs(
        self,
        args: Tuple,
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

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> SwitchTrace:
        (idx, *branch_args) = args
        self.static_check_num_arguments_equals_num_branches(branch_args)

        def _inner(idx: int):
            return lambda trace_leaves, retval_leaves, key, args: self._simulate(
                trace_leaves, retval_leaves, key, idx, args
            )

        branch_functions = list(map(_inner, range(len(self.branches))))
        (
            (trace_leaves, trace_defs),
            (retval_leaves, retval_defs),
        ) = self._empty_simulate_defs(tuple(branch_args))
        (trace_leaves, retval_leaves), score = jax.lax.switch(
            idx, branch_functions, trace_leaves, retval_leaves, key, tuple(branch_args)
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
        retval = Sum.maybe_none(idx, retvals)
        return SwitchTrace(self, args, subtraces, retval, score)

    @typecheck
    def _empty_update_defs(
        self,
        trace: SwitchTrace,
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ):
        trace_defs = []
        trace_leaves = []
        bwd_problem_defs = []
        bwd_problem_leaves = []
        retdiff_defs = []
        retdiff_leaves = []
        for static_idx in range(len(self.branches)):
            subtrace = trace.subtraces[static_idx]
            gen_fn = self.branches[static_idx]
            branch_argdiffs = argdiffs[static_idx]
            key = jax.random.PRNGKey(0)
            trace_shape, _, retdiff_shape, bwd_problem_shape = get_data_shape(
                gen_fn.update
            )(key, subtrace, problem, branch_argdiffs)
            empty_trace = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), trace_shape
            )
            empty_retdiff = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), retdiff_shape
            )
            empty_problem = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), bwd_problem_shape
            )
            trace_leaf, trace_def = jtu.tree_flatten(empty_trace)
            bwd_problem_leaf, bwd_problem_def = jtu.tree_flatten(empty_problem)
            retdiff_leaf, retdiff_def = jtu.tree_flatten(empty_retdiff)
            trace_defs.append(trace_def)
            trace_leaves.append(trace_leaf)
            bwd_problem_defs.append(bwd_problem_def)
            bwd_problem_leaves.append(bwd_problem_leaf)
            retdiff_defs.append(retdiff_def)
            retdiff_leaves.append(retdiff_leaf)
        return (
            (trace_leaves, trace_defs),
            (retdiff_leaves, retdiff_defs),
            (bwd_problem_leaves, bwd_problem_defs),
        )

    def _specialized_update_idx_no_change(
        self,
        key: PRNGKey,
        static_idx: Int,
        trace: SwitchTrace,
        problem: UpdateProblem,
        idx: IntArray,
        argdiffs: Argdiffs,
    ):
        subtrace = trace.subtraces[static_idx]
        gen_fn = self.branches[static_idx]
        branch_argdiffs = argdiffs[static_idx]
        tr, w, rd, bwd_problem = gen_fn.update(key, subtrace, problem, branch_argdiffs)
        (
            (trace_leaves, _),
            (retdiff_leaves, _),
            (bwd_problem_leaves, _),
        ) = self._empty_update_defs(trace, problem, argdiffs)
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retdiff_leaves[static_idx] = jtu.tree_leaves(rd)
        bwd_problem_leaves[static_idx] = jtu.tree_leaves(bwd_problem)
        score = tr.get_score()
        return (trace_leaves, retdiff_leaves, bwd_problem_leaves), (score, w)

    @typecheck
    def _generic_update_idx_change(
        self,
        key: PRNGKey,
        static_idx: Int,
        trace: SwitchTrace,
        problem: UpdateProblem,
        idx: IntArray,
        argdiffs: Argdiffs,
    ):
        gen_fn = self.branches[static_idx]
        branch_argdiffs = argdiffs[static_idx]
        check = static_idx == idx
        branch_primals = Diff.tree_primal(branch_argdiffs)
        new_subtrace = gen_fn.simulate(key, branch_primals)
        new_subtrace_def = jtu.tree_structure(new_subtrace)
        _, _, _, bwd_problem_shape = get_data_shape(gen_fn.update)(
            key, new_subtrace, problem, branch_argdiffs
        )
        bwd_problem_def = jtu.tree_structure(bwd_problem_shape)

        def _update_same_branch(key, subtrace, problem, branch_argdiffs):
            tr, w, rd, bwd_problem = gen_fn.update(
                key, subtrace, problem, branch_argdiffs
            )
            rd = Diff.tree_diff_unknown_change(rd)
            tr_leaves = jtu.tree_leaves(tr)
            problem_leaves = jtu.tree_leaves(bwd_problem)
            return tr_leaves, w, rd, problem_leaves

        def _update_new_branch(key, subtrace, problem, branch_argdiffs):
            branch_argdiffs = Diff.tree_diff_no_change(branch_argdiffs)
            tr, w, rd, bwd_problem = gen_fn.update(
                key, subtrace, problem, branch_argdiffs
            )
            rd = Diff.tree_diff_unknown_change(rd)
            tr_leaves = jtu.tree_leaves(tr)
            problem_leaves = jtu.tree_leaves(bwd_problem)
            return tr_leaves, w, rd, problem_leaves

        tr_leaves, w, rd, bwd_problem_leaves = jax.lax.cond(
            check,
            _update_same_branch,
            _update_new_branch,
            key,
            new_subtrace,
            problem,
            branch_argdiffs,
        )
        tr = jtu.tree_unflatten(new_subtrace_def, tr_leaves)
        bwd_problem = jtu.tree_unflatten(bwd_problem_def, bwd_problem_leaves)
        (
            (trace_leaves, _),
            (retdiff_leaves, _),
            (bwd_problem_leaves, _),
        ) = self._empty_update_defs(trace, problem, argdiffs)
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retdiff_leaves[static_idx] = jtu.tree_leaves(rd)
        bwd_problem_leaves[static_idx] = jtu.tree_leaves(bwd_problem)
        score = tr.get_score()
        return (trace_leaves, retdiff_leaves, bwd_problem_leaves), (score, w)

    @typecheck
    def update_generic(
        self,
        key: PRNGKey,
        trace: SwitchTrace,
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        (idx_argdiff, *branch_argdiffs) = argdiffs
        self.static_check_num_arguments_equals_num_branches(branch_argdiffs)

        def update_dispatch(static_idx: int):
            if Diff.tree_tangent(idx_argdiff) == NoChange:
                return (
                    lambda key,
                    trace,
                    problem,
                    idx,
                    argdiffs: self._specialized_update_idx_no_change(
                        key, static_idx, trace, problem, idx, argdiffs
                    )
                )
            else:
                return (
                    lambda key,
                    trace,
                    problem,
                    idx,
                    argdiffs: self._generic_update_idx_change(
                        key, static_idx, trace, problem, idx, argdiffs
                    )
                )

        primals = Diff.tree_primal(argdiffs)
        idx = primals[0]
        branch_functions = list(map(update_dispatch, range(len(self.branches))))

        (trace_leaves, retdiff_leaves, bwd_problem_leaves), (score, w) = jax.lax.switch(
            idx, branch_functions, key, trace, problem, idx, tuple(branch_argdiffs)
        )
        (
            (_, trace_defs),
            (_, retdiff_defs),
            (_, bwd_problem_defs),
        ) = self._empty_update_defs(trace, problem, tuple(branch_argdiffs))
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
        bwd_problems = list(
            map(
                lambda x: jtu.tree_unflatten(
                    bwd_problem_defs[x], bwd_problem_leaves[x]
                ),
                range(len(bwd_problem_leaves)),
            )
        )
        retdiff = Sum.maybe_none(idx_argdiff, retdiffs)
        retval = Diff.tree_primal(retdiff)
        if Diff.tree_tangent(idx_argdiff) == UnknownChange:
            w = w + (score - trace.get_score())

        return (
            SwitchTrace(self, primals, subtraces, retval, score),
            w,
            retdiff,
            SumProblem(idx, bwd_problems),
        )

    @typecheck
    def _empty_importance_defs(
        self,
        problem: ImportanceProblem,
        argdiffs: Argdiffs,
    ):
        trace_defs = []
        trace_leaves = []
        retval_defs = []
        retval_leaves = []
        bwd_problem_defs = []
        bwd_problem_leaves = []
        for static_idx in range(len(self.branches)):
            branch_gen_fn = self.branches[static_idx]
            branch_argdiffs = argdiffs[static_idx]
            key = jax.random.PRNGKey(0)
            trace_shape, _, _, bwd_problem_shape = get_data_shape(branch_gen_fn.update)(
                key,
                EmptyTrace(branch_gen_fn),
                problem,
                branch_argdiffs,
            )
            empty_trace = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), trace_shape
            )
            empty_problem = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), bwd_problem_shape
            )
            trace_leaf, trace_def = jtu.tree_flatten(empty_trace)
            retval_leaf, retval_def = jtu.tree_flatten(empty_trace.get_retval())
            bwd_problem_leaf, bwd_problem_def = jtu.tree_flatten(empty_problem)
            retval_defs.append(retval_def)
            retval_leaves.append(retval_leaf)
            trace_defs.append(trace_def)
            trace_leaves.append(trace_leaf)
            bwd_problem_defs.append(bwd_problem_def)
            bwd_problem_leaves.append(bwd_problem_leaf)
        return (
            (trace_leaves, trace_defs),
            (retval_leaves, retval_defs),
            (bwd_problem_leaves, bwd_problem_defs),
        )

    def _importance(
        self,
        trace_leaves,
        retval_leaves,
        bwd_problem_leaves,
        key,
        static_idx,
        constraint,
        argdiffs,
    ):
        branch_gen_fn = self.branches[static_idx]
        branch_argdiffs = argdiffs[static_idx]
        tr, w, _, bwd_problem = branch_gen_fn.update(
            key, EmptyTrace(branch_gen_fn), constraint, branch_argdiffs
        )
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retval_leaves[static_idx] = jtu.tree_leaves(tr.get_retval())
        bwd_problem_leaves[static_idx] = jtu.tree_leaves(bwd_problem)
        score = tr.get_score()
        return (trace_leaves, retval_leaves, bwd_problem_leaves), (score, w)

    @typecheck
    def update_importance(
        self,
        key: PRNGKey,
        problem: ImportanceProblem,
        argdiffs: Tuple,
    ) -> Tuple[SwitchTrace, Weight, Retdiff, UpdateProblem]:
        args = Diff.tree_primal(argdiffs)
        (idx, *branch_args) = args
        (_, *branch_argdiffs) = argdiffs
        branch_argdiffs = tuple(branch_argdiffs)
        self.static_check_num_arguments_equals_num_branches(branch_args)

        def _inner(static_idx: int):
            return (
                lambda trace_leaves,
                retval_leaves,
                bwd_problem_leaves,
                key,
                problem,
                branch_argdiffs: self._importance(
                    trace_leaves,
                    retval_leaves,
                    bwd_problem_leaves,
                    key,
                    static_idx,
                    problem,
                    branch_argdiffs,
                )
            )

        branch_functions = list(map(_inner, range(len(self.branches))))
        (
            (trace_leaves, trace_defs),
            (retval_leaves, retval_defs),
            (bwd_problem_leaves, bwd_problem_defs),
        ) = self._empty_importance_defs(problem, branch_argdiffs)

        (trace_leaves, retval_leaves, bwd_problem_leaves), (score, w) = jax.lax.switch(
            idx,
            branch_functions,
            trace_leaves,
            retval_leaves,
            bwd_problem_leaves,
            key,
            problem,
            branch_argdiffs,
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
        bwd_problems = list(
            map(
                lambda x: jtu.tree_unflatten(
                    bwd_problem_defs[x], bwd_problem_leaves[x]
                ),
                range(len(bwd_problem_leaves)),
            )
        )
        retval = Sum.maybe_none(idx, retvals)
        return (
            SwitchTrace(self, args, subtraces, retval, score),
            w,
            Diff.unknown_change(retval),
            SumProblem(idx, bwd_problems),
        )

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        assert isinstance(trace, EmptyTrace | SwitchTrace)
        match trace:
            case EmptyTrace():
                return self.update_importance(key, problem, argdiffs)
            case SwitchTrace():
                return self.update_generic(key, trace, problem, argdiffs)

    def _empty_assess_defs(self, sample: Sample, args: Tuple):
        retval_defs = []
        retval_leaves = []
        for static_idx in range(len(self.branches)):
            branch_gen_fn = self.branches[static_idx]
            _, retval_shape = get_data_shape(branch_gen_fn.assess)(sample, args)
            empty_retval = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype), retval_shape
            )
            retval_leaf, retval_def = jtu.tree_flatten(empty_retval)
            retval_defs.append(retval_def)
            retval_leaves.append(retval_leaf)
        return (retval_leaves, retval_defs)

    def _assess(self, static_idx, sample, args):
        branch_gen_fn = self.branches[static_idx]
        score, retval = branch_gen_fn.assess(sample, args)
        (retval_leaves, _) = self._empty_assess_defs(sample, args)
        retval_leaves[static_idx] = jtu.tree_leaves(retval)
        return retval_leaves, score

    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        (idx, *branch_args) = args
        self.static_check_num_arguments_equals_num_branches(branch_args)

        def _inner(static_idx: int):
            return lambda sample, args: self._assess(static_idx, sample, args)

        branch_functions = list(map(_inner, range(len(self.branches))))

        retval_leaves, score = jax.lax.switch(
            idx, branch_functions, sample, branch_args
        )
        (_, retval_defs) = self._empty_assess_defs(sample, args)
        retvals = list(
            map(
                lambda x: jtu.tree_unflatten(retval_defs[x], retval_leaves[x]),
                range(len(retval_leaves)),
            )
        )
        retval = Sum.maybe_none(idx, retvals)
        return score, retval


#############
# Decorator #
#############


@typecheck
def switch_combinator(
    *f: GenerativeFunction,
) -> SwitchCombinator:
    return SwitchCombinator(f)
