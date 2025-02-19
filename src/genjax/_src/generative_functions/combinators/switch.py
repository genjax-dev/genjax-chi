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


from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    Constraint,
    EditRequest,
    GenerativeFunction,
    Retdiff,
    Score,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.generative.choice_map import ExtendedAddress, Selection
from genjax._src.core.interpreters.incremental import Diff, NoChange, UnknownChange
from genjax._src.core.interpreters.staging import multi_switch, tree_choose
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    Generic,
    IntArray,
    PRNGKey,
    TypeVar,
)

R = TypeVar("R")

################
# Switch trace #
################


@Pytree.dataclass
class SwitchTrace(Generic[R], Trace[R]):
    gen_fn: "Switch[R]"
    args: tuple[Any, ...]
    subtraces: list[Trace[R]]
    retval: R
    score: FloatArray

    def get_idx(self) -> int | IntArray:
        """
        Get the index used to select the branch in this SwitchTrace.

        Returns:
            The index value used to select the executed branch.

        Note:
            This method assumes that the first argument passed to the Switch was the index used for branch selection.
        """
        return self.get_args()[0]

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_choices(self) -> ChoiceMap:
        idx = self.get_idx()
        sub_chms = (tr.get_choices() for tr in self.subtraces)
        return ChoiceMap.switch(idx, sub_chms)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_inner_trace(self, address: ExtendedAddress):
        assert isinstance(address, int)
        return self.subtraces[address]


#####################
# Switch combinator #
#####################


@Pytree.dataclass
class Switch(Generic[R], GenerativeFunction[R]):
    """
    `Switch` accepts `n` generative functions as input and returns a new [`genjax.GenerativeFunction`][] that accepts `n+1` arguments:

    - an index in the range `[0, n-1]`
    - a tuple of arguments for each of the input generative functions

    and executes the generative function at the supplied index with its provided arguments.

    If `index` is out of bounds, `index` is clamped to within bounds.

    !!! info "Existence uncertainty"

        This pattern allows `GenJAX` to express existence uncertainty over random choices -- as different generative function branches need not share addresses.

    Attributes:
        branches: generative functions that the `Switch` will select from based on the supplied index.

    Examples:
        Create a `Switch` via the [`genjax.switch`][] method:
        ```python exec="yes" html="true" source="material-block" session="switch"
        import jax, genjax


        @genjax.gen
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"


        @genjax.gen
        def branch_2():
            x = genjax.bernoulli(probs=0.3) @ "x2"


        switch = genjax.switch(branch_1, branch_2)

        key = jax.random.key(314159)
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
        return tree_choose(idx, retvals)

    def _check_args_match_branches(self, args):
        assert len(args) == len(self.branches)

    ## Simulate methods

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> SwitchTrace[R]:
        idx, branch_args = args[0], args[1:]
        self._check_args_match_branches(branch_args)

        fs = list(f.simulate for f in self.branches)
        f_args = list((key, args) for args in branch_args)

        subtraces = multi_switch(idx, fs, f_args)
        retval, score = tree_choose(
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

        return tree_choose(idx, multi_switch(idx, fs, f_args))

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

        pairs = multi_switch(idx, fs, f_args)
        subtraces = list(tr for tr, _ in pairs)

        retval, score, weight = tree_choose(
            idx, list((tr.get_retval(), tr.get_score(), w) for tr, w in pairs)
        )
        return SwitchTrace(self, args, subtraces, retval, score), weight

    def project(
        self,
        key: PRNGKey,
        trace: Trace[R],
        selection: Selection,
    ) -> Weight:
        assert isinstance(trace, SwitchTrace)
        idx = trace.get_idx()

        fs = list(f.project for f in self.branches)
        f_args = list((key, tr, selection) for tr in trace.subtraces)

        return tree_choose(idx, multi_switch(idx, fs, f_args))

    def _make_edit_fresh_trace(self, gen_fn: GenerativeFunction[R]):
        """
        Creates a function to handle editing a fresh trace when the switch index changes.

        This method is used internally by the `edit` method to handle cases where
        the switch index has changed, requiring the generation of a new trace
        for the selected branch.
        """

        def inner(
            key: PRNGKey,
            edit_request: Update,
            argdiffs: Argdiffs,
        ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
            # the old trace only has a filled-in subtrace for the original index. All other subtraces are filled with zeros. In the case of a changed index we need to
            #
            # - generate a fresh trace for the new branch,
            # - call `edit` with that new trace (setting the argdiffs passed into `edit` as `no_change`, since we used the same args to create the new trace)
            # - return the edit result with the `retdiff` wrapped in `unknown_change` (since our return value comes from a new branch)
            primals = Diff.tree_primal(argdiffs)
            new_trace = gen_fn.simulate(key, primals)

            tr, w, rd, bwd_request = gen_fn.edit(
                key,
                new_trace,
                edit_request,
                Diff.no_change(argdiffs),
            )
            return tr, w, Diff.unknown_change(rd), bwd_request

        return inner

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[SwitchTrace[R], Weight, Retdiff[R], EditRequest]:
        assert isinstance(edit_request, Update)
        assert isinstance(trace, SwitchTrace)

        idx_diff, branch_argdiffs = argdiffs[0], argdiffs[1:]
        self._check_args_match_branches(branch_argdiffs)

        primals = Diff.tree_primal(argdiffs)
        new_idx = primals[0]

        if Diff.tree_tangent(idx_diff) == NoChange:
            # If the index hasn't changed, perform edits on each branch.
            fs = list(f.edit for f in self.branches)
            f_args = list(
                (key, trace, edit_request, argdiffs)
                for trace, argdiffs in zip(trace.subtraces, branch_argdiffs)
            )
        else:
            fs = list(self._make_edit_fresh_trace(f) for f in self.branches)
            f_args = list((key, edit_request, argdiffs) for argdiffs in branch_argdiffs)

        rets = multi_switch(new_idx, fs, f_args)

        subtraces = list(t[0] for t in rets)
        score, weight, retdiff = tree_choose(
            new_idx, list((tr.get_score(), w, rd) for tr, w, rd, _ in rets)
        )
        retval: R = Diff.tree_primal(retdiff)

        if Diff.tree_tangent(idx_diff) == UnknownChange:
            weight += score - trace.get_score()

        # TODO: this is totally wrong, fix in future PR.
        bwd_request: Update = rets[0][3]

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
) -> Switch[R]:
    """
    Given `n` [`genjax.GenerativeFunction`][] inputs, returns a [`genjax.GenerativeFunction`][] that accepts `n+1` arguments:

    - an index in the range $[0, n)$
    - a tuple of arguments for each of the input generative functions (`n` total tuples)

    and executes the generative function at the supplied index with its provided arguments.

    If `index` is out of bounds, `index` is clamped to within bounds.

    Args:
        gen_fns: generative functions that the `Switch` will select from.

    Examples:
        Create a `Switch` via the [`genjax.switch`][] method:
        ```python exec="yes" html="true" source="material-block" session="switch"
        import jax, genjax


        @genjax.gen
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"


        @genjax.gen
        def branch_2():
            x = genjax.bernoulli(probs=0.3) @ "x2"


        switch = genjax.switch(branch_1, branch_2)

        key = jax.random.key(314159)
        jitted = jax.jit(switch.simulate)

        # Select `branch_2` by providing 1:
        tr = jitted(key, (1, (), ()))

        print(tr.render_html())
        ```
    """
    return Switch[R](gen_fns)
