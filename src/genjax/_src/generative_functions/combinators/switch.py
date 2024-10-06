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
    staged_choose,
    to_shape_fn,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    IntArray,
    Iterable,
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


def _wrapped(static_idx: int, f: Callable[..., Any]):
    def foo(shapes: list[R], arg_tuples) -> list[R]:
        shapes[static_idx] = f(*arg_tuples[static_idx])
        return shapes

    return foo


def _switch(
    idx, branches: Iterable[Callable[..., Any]], arg_tuples: Iterable[tuple[Any, ...]]
):
    "Returns a pivoted blah. tuples of the first, second, third etc retvals."
    shapes = list(
        to_shape_fn(f, jnp.zeros)(*args) for f, args in zip(branches, arg_tuples)
    )
    fns = list(_wrapped(i, f) for i, f in enumerate(branches))
    return jax.lax.switch(idx, fns, shapes, arg_tuples)


@Pytree.dataclass
class SwitchTrace(Generic[R], Trace[R]):
    gen_fn: "SwitchCombinator[R]"
    args: tuple[Any, ...]
    subtraces: list[Trace[R]]
    retval: R
    score: FloatArray

    def get_idx(self) -> IntArray:
        return self.get_args()[0]

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_choices(self) -> ChoiceMap:
        idx = self.get_idx()
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
        assert isinstance(trace, SwitchTrace)
        idx = trace.get_idx()

        fs = list(f.project for f in self.branches)
        f_args = list((key, tr, projection) for tr in trace.subtraces)

        weights = _switch(idx, fs, f_args)
        return staged_choose(idx, weights)

    def _make_edit_changed_idx(self, static_idx: int, gen_fn: GenerativeFunction[R]):
        gen_fn = self.branches[static_idx]

        def inner(
            idx: IntArray,
            key: PRNGKey,
            edit_request: IncrementalGenericRequest,
            argdiffs: Argdiffs,
        ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
            primals = Diff.tree_primal(argdiffs)
            new_trace = gen_fn.simulate(key, primals)

            branch_ad: Argdiffs = jax.lax.cond(
                static_idx == idx,
                lambda: argdiffs,
                lambda: Diff.no_change(argdiffs),
            )
            tr, w, rd, bwd_request = gen_fn.edit(
                key,
                new_trace,
                edit_request,
                branch_ad,
            )
            rd = Diff.unknown_change(rd)
            return tr, w, rd, bwd_request

        return inner

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[SwitchTrace[R], Weight, Retdiff[R], EditRequest]:
        assert isinstance(edit_request, IncrementalGenericRequest)
        assert isinstance(trace, SwitchTrace)

        idx_diff, branch_argdiffs = argdiffs[0], argdiffs[1:]
        self._check_args_match_branches(branch_argdiffs)

        primals = Diff.tree_primal(argdiffs)
        idx = primals[0]

        if Diff.tree_tangent(idx_diff) == NoChange:
            # If the index hasn't changed, perform edits on each branch.
            fs = list(f.edit for f in self.branches)
            f_args = list(
                (key, trace, edit_request, argdiffs)
                for trace, argdiffs in zip(trace.subtraces, branch_argdiffs)
            )
        else:
            fs = list(
                self._make_edit_changed_idx(i, f) for i, f in enumerate(self.branches)
            )
            f_args = list(
                (idx, key, edit_request, argdiffs) for argdiffs in branch_argdiffs
            )

        rets = _switch(idx, fs, f_args)

        subtraces = list(t[0] for t in rets)
        score, weight, retdiff = staged_choose(
            idx, list((tr.get_score(), w, rd) for tr, w, rd, _ in rets)
        )
        retval: R = Diff.tree_primal(retdiff)

        if Diff.tree_tangent(idx_diff) == UnknownChange:
            weight += score - trace.get_score()

        # TODO: this is totally wrong, fix in future PR.
        bwd_request: IncrementalGenericRequest = rets[0][3]

        return (
            SwitchTrace(self, primals, subtraces, retval, score),
            weight,
            retdiff,
            bwd_request,
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
