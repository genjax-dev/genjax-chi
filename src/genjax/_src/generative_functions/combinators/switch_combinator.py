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
    Constraint,
    GenerativeFunction,
    GenerativeFunctionClosure,
    Retdiff,
    Sample,
    SwitchConstraint,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    IntArray,
    List,
    PRNGKey,
    Sequence,
    Tuple,
    typecheck,
)

######################
# Switch sample type #
######################


@Pytree.dataclass
class SwitchSample(Sample):
    index: IntArray
    subtraces: Sequence[Sample]

    def get_constraint(self):
        return SwitchConstraint(
            self.index,
            list(map(lambda x: x.get_constraint(), self.subtraces)),
        )


################
# Switch trace #
################


@Pytree.dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    idx: IntArray
    subtraces: List[Trace]
    retval: Any
    score: FloatArray

    def get_sample(self):
        return SwitchSample(
            self.idx,
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
        ```python exec="yes" source="tabbed-left"
        import jax
        import genjax

        console = genjax.console()


        @genjax.static_gen_fn
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"


        @genjax.static_gen_fn
        def branch_2():
            x = genjax.bernoulli(0.3) @ "x2"


        ################################################################################
        # Creating a `SwitchCombinator` via the preferred `switch_combinator` function #
        ################################################################################

        switch = genjax.switch_combinator(branch_1, branch_2)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(switch.simulate)
        _ = jitted(key, (0,))
        tr = jitted(key, (1,))

        print(console.render(tr))
        ```
    """

    idx: IntArray
    branch_args: Tuple
    branches: Tuple[GenerativeFunctionClosure, ...]

    def get_branch_gen_fn(self, idx: int):
        branch_closure = self.branches[idx]
        branch_args = self.branch_args[idx]
        return (
            branch_closure(*branch_args)
            if isinstance(branch_args, tuple)
            else branch_closure(branch_args)
        )

    # Optimized abstract call for tracing.
    def __abstract_call__(self):
        gen_fn = self.get_branch_gen_fn(0)
        return gen_fn.__abstract_call__()

    def _empty_trace_leaves(self):
        trace_leaves = []
        for idx in range(len(self.branches)):
            gen_fn = self.get_branch_gen_fn(idx)
            empty_trace = gen_fn.get_empty_trace()
            leaves = jtu.tree_leaves(empty_trace)
            trace_leaves.append(leaves)
        return trace_leaves

    def _empty_trace_defs(self):
        trace_defs = []
        for idx in range(len(self.branches)):
            gen_fn = self.get_branch_gen_fn(idx)
            empty_trace = gen_fn.get_empty_trace()
            trace_def = jtu.tree_structure(empty_trace)
            trace_defs.append(trace_def)
        return trace_defs

    def _simulate(self, key, static_idx):
        branch_gen_fn = self.get_branch_gen_fn(static_idx)
        tr = branch_gen_fn.simulate(key)
        trace_leaves = self._empty_trace_leaves()
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retval = tr.get_retval()
        score = tr.get_score()
        return trace_leaves, (score, retval)

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
    ) -> SwitchTrace:
        def _inner(idx: int):
            return lambda key: self._simulate(key, idx)

        branch_functions = list(map(_inner, range(len(self.branches))))
        trace_defs = self._empty_trace_defs()
        trace_leaves, (score, retval) = jax.lax.switch(self.idx, branch_functions, key)
        subtraces = list(
            map(
                lambda x: jtu.tree_unflatten(trace_defs[x], trace_leaves[x]),
                range(len(trace_leaves)),
            )
        )
        return SwitchTrace(self, self.idx, subtraces, retval, score)

    def _importance(self, key, branch, static_idx, constraint):
        branch_gen_fn = branch(*self.branch_args)
        tr, w = branch_gen_fn.importance(key, constraint)
        trace_leaves = self._empty_trace_leaves()
        trace_leaves[static_idx] = jtu.tree_leaves(tr)
        retval = tr.get_retval()
        score = tr.get_score()
        return trace_leaves, (score, retval), w

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ) -> Tuple[SwitchTrace, Weight]:
        def _inner(idx: int, branch):
            return lambda key, constraint: self._importance(
                key, branch, idx, constraint
            )

        branch_functions = list(
            map(lambda args: _inner(args[0], args[1]), enumerate(self.branches))
        )

        trace_leaves, (score, retval), w = jax.lax.switch(
            self.idx, branch_functions, key, constraint
        )
        trace_defs = self._empty_trace_defs()
        trace_leaves, (score, retval) = jax.lax.switch(self.idx, branch_functions, key)
        subtraces = list(
            map(
                lambda x: jtu.tree_unflatten(trace_defs[x], trace_leaves[x]),
                range(len(trace_leaves)),
            )
        )
        return SwitchTrace(self, self.idx, subtraces, retval, score), w

    def update(
        self,
        key: PRNGKey,
        prev: SwitchTrace,
        spec: UpdateSpec,
    ) -> Tuple[SwitchTrace, Weight, Retdiff, UpdateSpec]:
        pass

    @typecheck
    def assess(
        self,
        constraint: Constraint,
    ) -> Tuple[FloatArray, Any]:
        pass


#############
# Decorator #
#############


def switch_combinator(
    *f: GenerativeFunction,
) -> GenerativeFunctionClosure:
    @GenerativeFunction.closure
    def inner(idx: IntArray, *args: Any) -> SwitchCombinator:
        return SwitchCombinator(idx, args, f)

    return inner
