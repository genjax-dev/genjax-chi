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


import functools

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
from genjax._src.core.interpreters.incremental import Diff, NoChange, UnknownChange
from genjax._src.core.interpreters.staging import (
    empty_edit,
    staged_choose,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
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
def _eval_zero(f: Callable[..., Any], *args, **kwargs):
    shape = jax.eval_shape(f, *args, **kwargs)
    return jtu.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), shape)


def _wrapped(static_idx: int, f: Callable[..., Any]):
    def foo(shapes: list[R], arg_tuples) -> list[R]:
        shapes[static_idx] = f(*arg_tuples[static_idx])
        return shapes

    return foo


def _switch(
    idx, branches: Iterable[Callable[..., Any]], arg_tuples: Iterable[tuple[Any, ...]]
):
    "Returns a pivoted blah. tuples of the first, second, third etc retvals."
    shapes = list(_eval_zero(f, *args) for f, args in zip(branches, arg_tuples))
    fns = list(_wrapped(i, f) for i, f in enumerate(branches))
    return jax.lax.switch(idx, fns, shapes, arg_tuples)


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

    def _indices(self):
        return range(len(self.branches))

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
    #
    # TODO is it a bug that we reuse the same key?
    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> SwitchTrace[R]:
        idx, branch_args = args[0], args[1:]
        self._check_args_match_branches(branch_args)

        fs = list(f.simulate for f in self.branches)
        f_args = list((key, args) for args in branch_args)

        subtraces = _switch(idx, fs, f_args)
        retval, score = staged_choose(
            idx, list((tr.get_retval(), tr.get_score()) for tr in subtraces)
        )
        return SwitchTrace(self, args, subtraces, retval, score)

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        idx, branch_args = args[0], args[1:]
        self._check_args_match_branches(branch_args)

        fs = list(f.assess for f in self.branches)
        f_args = list((sample, args) for args in branch_args)

        return staged_choose(idx, _switch(idx, fs, f_args))

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[SwitchTrace[R], Weight]:
        idx, branch_args = args[0], args[1:]
        self._check_args_match_branches(branch_args)

        fs = list(f.generate for f in self.branches)
        f_args = list((key, constraint, args) for args in branch_args)

        pairs = _switch(idx, fs, f_args)
        subtraces = list(tr for tr, _ in pairs)

        retval, score, weight = staged_choose(
            idx, list((tr.get_retval(), tr.get_score(), w) for tr, w in pairs)
        )
        return SwitchTrace(self, args, subtraces, retval, score), weight

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
        constraint: IncrementalGenericRequest,
        argdiffs: tuple[Argdiffs, ...],
    ):
        def _unpack(f, tr, diffs):
            empty_tr, _, empty_retdiff, empty_problem = empty_edit(
                f, tr, constraint, diffs
            )
            return empty_tr, empty_retdiff, empty_problem

        return self._to_pairs(
            _unpack(f, tr, diffs)
            for f, tr, diffs in zip(self.branches, trace.subtraces, argdiffs)
        )

    def _generic_edit_idx_change(
        self,
        static_idx: Int,
        key: PRNGKey,
        trace: SwitchTrace[R],
        constraint: IncrementalGenericRequest,
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
            constraint,
            branch_argdiffs,
        )
        bwd_request_def = jtu.tree_structure(empty_bwd_request)

        def _edit_same_branch(key, subtrace, constraint, branch_argdiffs):
            tr, w, rd, bwd_request = gen_fn.edit(
                key,
                subtrace,
                constraint,
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
                constraint,
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
        constraint: IncrementalGenericRequest,
        argdiffs: Argdiffs,
    ) -> tuple[SwitchTrace[R], Weight, Retdiff[R], EditRequest]:
        idx_argdiff, branch_argdiffs = argdiffs[0], argdiffs[1:]
        self._check_args_match_branches(branch_argdiffs)

        primals = Diff.tree_primal(argdiffs)
        idx = primals[0]

        if Diff.tree_tangent(idx_argdiff) == NoChange:
            fs = list(f.edit for f in self.branches)
            f_args = list(
                (key, subtrace, constraint, argdiffs)
                for subtrace, argdiffs in zip(trace.subtraces, branch_argdiffs)
            )
            rets = _switch(idx, fs, f_args)

            subtraces = list(t[0] for t in rets)
            # TODO: this is totally wrong, fix in future PR.
            bwd_request: IncrementalGenericRequest = rets[0][3]

            score, weight, retdiff = staged_choose(
                idx, list((tr.get_score(), w, rd) for tr, w, rd, _ in rets)
            )
            retval: R = Diff.tree_primal(retdiff)

            return (
                SwitchTrace(self, primals, subtraces, retval, score),
                weight,
                retdiff,
                bwd_request,
            )

        else:
            pass

        def edit_dispatch(static_idx: int):
            return functools.partial(self._generic_edit_idx_change, static_idx)

        fs = list(edit_dispatch(i) for i in range(len(self.branches)))

        (trace_leaves, retdiff_leaves, bwd_request_leaves), (score, w) = jax.lax.switch(
            idx, fs, key, trace, constraint, idx, tuple(branch_argdiffs)
        )
        (
            (_, trace_defs),
            (_, retdiff_defs),
            (_, bwd_request_defs),
        ) = self._empty_edit_defs(trace, constraint, branch_argdiffs)

        subtraces = self._unflatten(trace_defs, trace_leaves)
        retdiffs = self._unflatten(retdiff_defs, retdiff_leaves)
        bwd_requests = self._unflatten(bwd_request_defs, bwd_request_leaves)

        retdiff: R = staged_choose(idx, retdiffs)
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
            edit_request,
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
