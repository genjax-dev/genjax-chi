# Copyright 2023 MIT Probabilistic Computing Project
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
"""This module implements a generative function combinator which allows
branching control flow for combinations of generative functions which can
return different shaped choice maps.

It's based on encoding a trace sum type using JAX - to bypass restrictions from `jax.lax.switch`_.

Generative functions which are passed in as branches to `SwitchCombinator`
must accept the same argument types, and return the same type of return value.

The internal choice maps for the branch generative functions
can have different shape/dtype choices. The resulting `SwitchTrace` will efficiently share `(shape, dtype)` storage across branches.

.. _jax.lax.switch: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html
"""

from dataclasses import dataclass

import jax

from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import LanguageConstructor
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import mask
from genjax._src.core.interpreters.incremental import static_check_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.interpreters.incremental import tree_diff_unknown_change
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.staging_utils import (
    get_discard_data_shape,
)
from genjax._src.generative_functions.combinators.staging_utils import (
    get_trace_data_shape,
)
from genjax._src.generative_functions.combinators.switch.sumtree import (
    DataSharedSumTree,
)
from genjax._src.generative_functions.combinators.switch.switch_datatypes import (
    SwitchChoiceMap,
)
from genjax._src.generative_functions.combinators.switch.switch_datatypes import (
    SwitchTrace,
)
from genjax._src.generative_functions.static.static_gen_fn import SupportsCalleeSugar


#####
# SwitchCombinator
#####


@dataclass
class SwitchCombinator(JAXGenerativeFunction, SupportsCalleeSugar):
    """> `SwitchCombinator` accepts multiple generative functions as input and
    implements `GenerativeFunction` interface semantics that support branching
    control flow patterns, including control flow patterns which branch on
    other stochastic choices.

    !!! info "Existence uncertainty"

        This pattern allows `GenJAX` to express existence uncertainty over random choices -- as different generative function branches need not share addresses.

    Examples:

        ```python exec="yes" source="tabbed-left"
        import jax
        import genjax
        console = genjax.pretty()

        @genjax.lang
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"

        @genjax.lang
        def branch_2():
            x = genjax.bernoulli(0.3) @ "x2"

        # Creating a `SwitchCombinator` via the preferred `new` class method.
        switch = genjax.SwitchCombinator.new(branch_1, branch_2)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(genjax.simulate(switch))
        _ = jitted(key, (0, ))
        tr = jitted(key, (1, ))

        print(console.render(tr))
        ```
    """

    branches: List[JAXGenerativeFunction]

    def flatten(self):
        return (self.branches,), ()

    @typecheck
    @classmethod
    def new(cls, *args: JAXGenerativeFunction) -> "SwitchCombinator":
        """The preferred constructor for `SwitchCombinator` generative function
        instances. The shorthand symbol is `Switch = SwitchCombinator.new`.

        Arguments:
            *args: JAX generative functions which will act as branch callees for the invocation of branching control flow.

        Returns:
            instance: A `SwitchCombinator` instance.
        """
        return SwitchCombinator([*args])

    # Optimized abstract call for tracing.
    def __abstract_call__(self, branch, *args):
        first_branch = self.branches[0]
        return first_branch.__abstract_call__(*args)

    # Method is used to create a branch-agnostic type
    # which is acceptable for JAX's typing across `lax.switch`
    # branches.
    def _create_data_shared_sum_tree_trace(self, key, tr, args):
        covers = []
        sub_args = args[1:]
        for gen_fn in self.branches:
            trace_shape = get_trace_data_shape(gen_fn, key, sub_args)
            covers.append(trace_shape)
        return DataSharedSumTree.new(tr, covers)

    def _create_data_shared_sum_tree_discard(
        self, key, discard, tr, constraints, argdiffs
    ):
        covers = []
        sub_argdiffs = argdiffs[1:]
        for idx, gen_fn in enumerate(self.branches):
            subtrace = tr.get_subtrace(idx)
            discard_shape = get_discard_data_shape(
                gen_fn, key, subtrace, constraints, sub_argdiffs
            )
            covers.append(discard_shape)
        return DataSharedSumTree.new(discard, covers)

    def _simulate(self, branch_gen_fn, key, args):
        tr = branch_gen_fn.simulate(key, args[1:])
        data_shared_sum_tree = self._create_data_shared_sum_tree_trace(key, tr, args)
        choices = list(data_shared_sum_tree.materialize_iterator())
        branch_index = args[0]
        choice_map = SwitchChoiceMap(branch_index, choices)
        score = tr.get_score()
        retval = tr.get_retval()
        trace = SwitchTrace(self, choice_map, args, retval, score)
        return trace

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> SwitchTrace:
        switch = args[0]

        def _inner(br):
            return lambda key, *args: self._simulate(br, key, args)

        branch_functions = list(map(_inner, self.branches))
        return jax.lax.switch(switch, branch_functions, key, *args)

    def _importance(self, branch_gen_fn, key, chm, args):
        (tr, w) = branch_gen_fn.importance(key, chm, args[1:])
        data_shared_sum_tree = self._create_data_shared_sum_tree_trace(key, tr, args)
        choices = list(data_shared_sum_tree.materialize_iterator())
        branch_index = args[0]
        choice_map = SwitchChoiceMap(branch_index, choices)
        score = tr.get_score()
        retval = tr.get_retval()
        trace = SwitchTrace(self, choice_map, args, retval, score)
        return (trace, w)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: Choice,
        args: Tuple,
    ) -> Tuple[SwitchTrace, FloatArray]:
        switch = args[0]

        def _inner(br):
            return lambda key, chm, *args: self._importance(br, key, chm, args)

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(switch, branch_functions, key, chm, *args)

    def _update_fallback(
        self,
        key: PRNGKey,
        prev: Trace,
        constraints: Choice,
        argdiffs: Tuple,
    ):
        def _inner_update(br, key):
            # Get the branch index (at tracing time) and use the branch
            # to update.
            concrete_branch_index = self.branches.index(br)

            # Run the update for this branch.
            prev_subtrace = prev.get_subtrace(concrete_branch_index)
            (tr, w, retval_diff, discard) = br.update(
                key, prev_subtrace, constraints, argdiffs[1:]
            )

            # Here, we create a DataSharedSumTree -- and we place the real trace
            # data inside of it.
            args = tree_diff_primal(argdiffs)
            data_shared_sum_tree = self._create_data_shared_sum_tree_trace(
                key, tr, args
            )
            choices = list(data_shared_sum_tree.materialize_iterator())
            choice_map = SwitchChoiceMap(concrete_branch_index, choices)

            # Here, we create a DataSharedSumTree -- and we place the real discard
            # data inside of it.
            data_shared_sum_tree = self._create_data_shared_sum_tree_discard(
                key, discard, prev, constraints, argdiffs
            )
            discard_choices = list(data_shared_sum_tree.materialize_iterator())
            discard = SwitchChoiceMap(concrete_branch_index, discard_choices)

            # Get all the metadata for update from the trace.
            score = tr.get_score()
            retval = tr.get_retval()
            trace = SwitchTrace(self, choice_map, args, retval, score)
            return (trace, w, retval_diff, discard)

        def _inner(br):
            return lambda key: _inner_update(br, key)

        branch_functions = list(map(_inner, self.branches))
        switch = tree_diff_primal(argdiffs[0])

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
        )

    def _update_branch_switch(
        self,
        key: PRNGKey,
        prev: Trace,
        constraints: Choice,
        argdiffs: Tuple,
    ):
        def _inner_importance(br, key, prev, constraints, argdiffs):
            concrete_branch_index = self.branches.index(br)
            stripped = prev.strip()
            constraints = stripped.unsafe_merge(constraints)
            args = tree_diff_primal(argdiffs)
            (tr, w) = br.importance(key, constraints, args[1:])
            update_weight = w - prev.get_score()
            discard = mask(True, stripped)
            retval = tr.get_retval()
            retval_diff = tree_diff_unknown_change(retval)

            # Here, we create a DataSharedSumTree -- and we place the real trace
            # data inside of it.
            data_shared_sum_tree = self._create_data_shared_sum_tree_trace(
                key, tr, args
            )
            choices = list(data_shared_sum_tree.materialize_iterator())
            choice_map = SwitchChoiceMap(concrete_branch_index, choices)

            # Get all the metadata for update from the trace.
            score = tr.get_score()
            trace = SwitchTrace(self, choice_map, args, retval, score)
            return (trace, update_weight, retval_diff, discard)

        def _inner(br):
            return lambda key, prev, constraints, argdiffs: _inner_importance(
                br, key, prev, constraints, argdiffs
            )

        branch_functions = list(map(_inner, self.branches))
        switch = tree_diff_primal(argdiffs[0])

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            prev,
            constraints,
            argdiffs,
        )

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: SwitchTrace,
        constraints: Choice,
        argdiffs: Tuple,
    ) -> Tuple[SwitchTrace, FloatArray, Any, Any]:
        index_argdiff = argdiffs[0]

        if static_check_no_change(index_argdiff):
            return self._update_fallback(key, prev, constraints, argdiffs)
        else:
            return self._update_branch_switch(key, prev, constraints, argdiffs)

    @typecheck
    def assess(
        self,
        chm: Choice,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        switch = args[0]

        def _assess(branch_gen_fn, chm, args):
            return branch_gen_fn.assess(chm, args[1:])

        def _inner(br):
            return lambda chm, *args: _assess(br, chm, args)

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(switch, branch_functions, chm, *args)


#########################
# Language constructors #
#########################


@typecheck
def switch_combinator(*gen_fn: JAXGenerativeFunction):
    return SwitchCombinator.new(*gen_fn)


Switch = LanguageConstructor(
    switch_combinator,
)
