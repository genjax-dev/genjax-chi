# Copyright 2022 MIT Probabilistic Computing Project
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
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.masking import mask
from genjax._src.core.interpreters.staging import get_trace_data_shape
from genjax._src.core.pytree import Sumtree
from genjax._src.core.transforms.incremental import UnknownChange
from genjax._src.core.transforms.incremental import diff
from genjax._src.core.transforms.incremental import static_check_is_diff
from genjax._src.core.transforms.incremental import static_check_no_change
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import SupportsBuiltinSugar
from genjax._src.generative_functions.combinators.switch.switch_datatypes import (
    SwitchChoiceMap,
)
from genjax._src.generative_functions.combinators.switch.switch_datatypes import (
    SwitchTrace,
)
from genjax._src.generative_functions.combinators.switch.switch_tracetypes import (
    SumTraceType,
)


#####
# SwitchCombinator
#####


@dataclass
class SwitchCombinator(JAXGenerativeFunction, SupportsBuiltinSugar):
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

        @genjax.gen
        def branch_1():
            x = genjax.normal(0.0, 1.0) @ "x1"

        @genjax.gen
        def branch_2():
            x = genjax.bernoulli(0.3) @ "x2"

        # Creating a `SwitchCombinator` via the preferred `new` class method.
        switch = genjax.SwitchCombinator.new(branch_1, branch_2)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(genjax.simulate(switch))
        key, _ = jitted(key, (0, ))
        key, tr = jitted(key, (1, ))

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
    def _create_sum_pytree(self, key, choices, args):
        covers = []
        for gen_fn in self.branches:
            trace_shape = get_trace_data_shape(gen_fn, key, args)
            covers.append(trace_shape)
        return Sumtree.new(choices, covers)

    def get_trace_type(self, *args):
        subtypes = []
        for gen_fn in self.branches:
            subtypes.append(gen_fn.get_trace_type(*args[1:]))
        return SumTraceType(subtypes)

    def _simulate(self, branch_gen_fn, key, args):
        key, tr = branch_gen_fn.simulate(key, args[1:])
        sum_pytree = self._create_sum_pytree(key, tr, args[1:])
        choices = list(sum_pytree.materialize_iterator())
        branch_index = args[0]
        choice_map = SwitchChoiceMap(branch_index, choices)
        score = tr.get_score()
        retval = tr.get_retval()
        trace = SwitchTrace(self, choice_map, args, retval, score)
        return key, trace

    def simulate(self, key, args):
        switch = args[0]

        def _inner(br):
            return lambda key, *args: self._simulate(br, key, args)

        branch_functions = list(map(_inner, self.branches))
        return jax.lax.switch(switch, branch_functions, key, *args)

    def _importance(self, branch_gen_fn, key, chm, args):
        key, (w, tr) = branch_gen_fn.importance(key, chm, args[1:])
        sum_pytree = self._create_sum_pytree(key, tr, args[1:])
        choices = list(sum_pytree.materialize_iterator())
        branch_index = args[0]
        choice_map = SwitchChoiceMap(branch_index, choices)
        score = tr.get_score()
        retval = tr.get_retval()
        trace = SwitchTrace(self, choice_map, args, retval, score)
        return key, (w, trace)

    def importance(self, key, chm, args):
        switch = args[0]

        def _inner(br):
            return lambda key, chm, *args: self._importance(br, key, chm, args)

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(switch, branch_functions, key, chm, *args)

    def _update_fallback(
        self,
        key: PRNGKey,
        prev: Trace,
        new: ChoiceMap,
        argdiffs: Tuple,
    ):
        def _inner_update(br, key, prev, new, argdiffs):
            # Create a skeleton discard instance.
            discard_option = mask(False, prev.strip())
            concrete_branch_index = self.branches.index(br)

            prev_subtrace = prev.get_subtrace(concrete_branch_index)
            key, (retval_diff, w, tr, maybe_discard) = br.update(
                key, prev_subtrace, new, argdiffs[1:]
            )

            # Here, we create a Sumtree -- and we place the real trace
            # data inside of it.
            args = jtu.tree_map(
                tree_diff_primal, argdiffs, is_leaf=static_check_is_diff
            )
            sum_pytree = self._create_sum_pytree(key, tr, args[1:])
            choices = list(sum_pytree.materialize_iterator())
            choice_map = SwitchChoiceMap(concrete_branch_index, choices)

            # Merge the skeleton discard with the actual one.
            discard_option.submaps[concrete_branch_index] = maybe_discard

            # Get all the metadata for update from the trace.
            score = tr.get_score()
            retval = tr.get_retval()
            trace = SwitchTrace(self, choice_map, args, retval, score)
            return key, (retval_diff, w, trace, discard_option)

        def _inner(br):
            return lambda key, prev, new, argdiffs: _inner_update(
                br, key, prev, new, argdiffs
            )

        branch_functions = list(map(_inner, self.branches))
        switch = tree_diff_primal(argdiffs[0])

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            prev,
            new,
            argdiffs,
        )

    def _update_branch_switch(
        self,
        key: PRNGKey,
        prev: Trace,
        new: ChoiceMap,
        argdiffs: Tuple,
    ):
        def _inner_importance(br, key, prev, new, argdiffs):
            concrete_branch_index = self.branches.index(br)
            new = prev.strip().unsafe_merge(new)
            args = tree_diff_primal(argdiffs)
            key, (w, tr) = br.importance(key, new, args[1:])
            update_weight = w - prev.get_score()
            discard = mask(True, prev.strip())
            retval = tr.get_retval()
            retdiff = diff(retval, UnknownChange)

            sum_pytree = self._create_sum_pytree(key, tr, args[1:])
            choices = list(sum_pytree.materialize_iterator())
            choice_map = SwitchChoiceMap(concrete_branch_index, choices)

            # Get all the metadata for update from the trace.
            score = tr.get_score()
            trace = SwitchTrace(self, choice_map, args, retval, score)
            return key, (retdiff, update_weight, trace, discard)

        def _inner(br):
            return lambda key, prev, new, argdiffs: _inner_importance(
                br, key, prev, new, argdiffs
            )

        branch_functions = list(map(_inner, self.branches))
        switch = tree_diff_primal(argdiffs[0])

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            prev,
            new,
            argdiffs,
        )

    def update(self, key, prev, new, argdiffs):
        index_argdiff = argdiffs[0]

        if static_check_no_change(index_argdiff):
            return self._update_fallback(key, prev, new, argdiffs)
        else:
            return self._update_branch_switch(key, prev, new, argdiffs)

    def assess(self, key, chm, args):
        switch = args[0]

        def _assess(branch_gen_fn, key, chm, args):
            return branch_gen_fn.assess(key, chm, args[1:])

        def _inner(br):
            return lambda key, chm, *args: _assess(br, key, chm, args)

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(switch, branch_functions, key, chm, *args)


##############
# Shorthands #
##############

Switch = SwitchCombinator.new
