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
from genjax._src.core.interpreters.incremental import Diff, NoChange, UnknownChange
from genjax._src.core.interpreters.staging import (
    empty_assess,
    empty_edit,
    empty_generate,
    staged_choose,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    FloatArray,
    Generic,
    Int,
    IntArray,
    Iterable,
    PRNGKey,
    Sequence,
    TypeVar,
)

R = TypeVar("R")
_Leaf = Any
_UnflatPair = tuple[list[list[_Leaf]], list[jtu.PyTreeDef]]
_SingleUnflat = tuple[list[_Leaf], jtu.PyTreeDef]

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
        idx = self.get_args()[0]
        sub_chms = (tr.get_choices() for tr in self.subtraces)
        return ChoiceMap.switch(idx, sub_chms)

    def get_sample(self) -> ChoiceMap:
        return self.get_choices()

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

    @staticmethod
    def _to_pairs(seq_of_vs: Iterable[Sequence[Any]]) -> tuple[_UnflatPair, ...]:
        """Takes

        - a sequence of a sequence of pytrees

        And returns a sequence of pairs of

        - list of
        """
        acc: list[list[_SingleUnflat]] = []
        for vs in seq_of_vs:
            pairs: list[_SingleUnflat] = []
            for v in vs:
                pairs.append(jtu.tree_flatten(v))
            acc.append(pairs)

        print(acc)

        # *
        def pivot(*pairs: _SingleUnflat) -> _UnflatPair:
            """
            - pairs is called with the first `_SingleUnflat` from all entries
            - then all second `_SingleUnflat`
            - etc...

            for the nth call it returns a 2-tuple of

            - a list of the nth list[Leaf] from all entries
            - a list the nth PyTreeDef from all entries

            i.e., an `_UnflatPair`.
            """
            return tuple(map(list, zip(*pairs)))  # pyright: ignore

        return tuple(map(pivot, *acc))

    @staticmethod
    def _unflatten(defs: list[jtu.PyTreeDef], leaves: list[list[_Leaf]]) -> list[Any]:
        """Given the components of an `UnflatPair`, rebuilds the original objects."""
        return list(jtu.tree_unflatten(d, leaf) for d, leaf in zip(defs, leaves))

    def __abstract_call__(self, *args) -> R:
        idx, args = args[0], args[1:]
        retvals = list(
            f.__abstract_call__(*f_args) for f, f_args in zip(self.branches, args)
        )
        return staged_choose(idx, retvals)

    def _check_args_match_branches(self, args):
        assert len(args) == len(self.branches)

    ## Simulate methods

    def _empty_simulate_defs(
        self,
        arg_tuples: tuple[tuple[Any, ...], ...],
    ):
        def _unpack(f, args):
            empty_trace = f.get_zero_trace(*args)
            return empty_trace, empty_trace.get_retval()

        return self._to_pairs(
            _unpack(f, args) for f, args in zip(self.branches, arg_tuples)
        )

    def _simulate(self, key, static_idx, args, trace_leaves, retval_leaves):
        """
        This gets run for any particular switch, and populates the correct leaf.
        """
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

        self._check_args_match_branches(branch_args)

        def _inner(idx: int):
            return lambda key, args, trace_leaves, retval_leaves: self._simulate(
                key, idx, args, trace_leaves, retval_leaves
            )

        branch_functions = list(map(_inner, range(len(self.branches))))
        (
            (trace_leaves, trace_defs),
            (retval_leaves, retval_defs),
        ) = self._empty_simulate_defs(branch_args)

        (trace_leaves, retval_leaves), score = jax.lax.switch(
            idx, branch_functions, key, branch_args, trace_leaves, retval_leaves
        )
        subtraces = self._unflatten(trace_defs, trace_leaves)

        retvals: list[R] = self._unflatten(retval_defs, retval_leaves)
        retval: R = staged_choose(idx, retvals)
        return SwitchTrace(self, args, subtraces, retval, score)

    def _empty_assess_defs(
        self, sample: ChoiceMap, arg_tuples: tuple[tuple[Any, ...], ...]
    ):
        def _unpack(f, branch_args) -> tuple[R]:
            return (empty_assess(f, sample, branch_args)[1],)

        return self._to_pairs(
            _unpack(f, branch_args) for f, branch_args in zip(self.branches, arg_tuples)
        )

    def _assess(self, static_idx, sample, args):
        branch_gen_fn = self.branches[static_idx]
        branch_args = args[static_idx]
        score, retval = branch_gen_fn.assess(sample, branch_args)
        ((retval_leaves, _),) = self._empty_assess_defs(sample, args)
        retval_leaves[static_idx] = jtu.tree_leaves(retval)
        return retval_leaves, score

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        idx, branch_args = args[0], args[1:]
        self._check_args_match_branches(branch_args)

        def _inner(static_idx: int):
            return lambda sample, args: self._assess(static_idx, sample, args)

        branch_functions = list(map(_inner, range(len(self.branches))))

        retval_leaves, score = jax.lax.switch(
            idx, branch_functions, sample, branch_args
        )
        ((_, retval_defs),) = self._empty_assess_defs(sample, branch_args)
        retvals = self._unflatten(retval_defs, retval_leaves)
        retval: R = staged_choose(idx, retvals)
        return score, retval

    def _empty_generate_defs(
        self,
        constraint: Constraint,
        args: tuple[tuple[Any, ...], ...],
    ):
        def _unpack(f, args):
            empty_trace, _ = empty_generate(f, constraint, args)
            return empty_trace, empty_trace.get_retval()

        return self._to_pairs(_unpack(f, args) for f, args in zip(self.branches, args))

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
        self._check_args_match_branches(branch_args)

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
        subtraces = self._unflatten(trace_defs, trace_leaves)
        retvals = self._unflatten(retval_defs, retval_leaves)
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
        argdiffs: tuple[Argdiffs, ...],
    ):
        def _unpack(f, tr, diffs):
            empty_tr, _, empty_retdiff, empty_problem = empty_edit(
                f, tr, IncrementalGenericRequest(constraint), diffs
            )
            return empty_tr, empty_retdiff, empty_problem

        return self._to_pairs(
            _unpack(f, tr, diffs)
            for f, tr, diffs in zip(self.branches, trace.subtraces, argdiffs)
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
        _, _, _, empty_bwd_request = empty_edit(
            gen_fn,
            new_subtrace,
            IncrementalGenericRequest(constraint),
            branch_argdiffs,
        )
        bwd_request_def = jtu.tree_structure(empty_bwd_request)

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
        self._check_args_match_branches(branch_argdiffs)

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

        subtraces = self._unflatten(trace_defs, trace_leaves)
        retdiffs = self._unflatten(retdiff_defs, retdiff_leaves)
        bwd_requests = self._unflatten(bwd_request_defs, bwd_request_leaves)

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
