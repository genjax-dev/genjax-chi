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

Generative functions which are passed in as branches to :code:`SwitchCombinator`
must accept the same argument types, and return the same type of return value.

The internal choice maps for the branch generative functions
can have different shape/dtype choices. The resulting :code:`SwitchTrace` will efficiently share :code:`(shape, dtype)` storage across branches.

.. _jax.lax.switch: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html
"""

from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import Selection
from genjax._src.core.datatypes import Trace
from genjax._src.core.diff_rules import check_is_diff
from genjax._src.core.diff_rules import strip_diff
from genjax._src.core.masks import BooleanMask
from genjax._src.core.staging import get_trace_data_shape
from genjax._src.core.sumtree import Sumtree
from genjax._src.core.typing import FloatArray
from genjax._src.generative_functions.combinators.combinator_datatypes import (
    IndexedChoiceMap,
)
from genjax._src.generative_functions.combinators.combinator_tracetypes import (
    SumTraceType,
)


#####
# SwitchTrace
#####


@dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    chm: IndexedChoiceMap
    args: Tuple
    retval: Any
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.chm,
            self.args,
            self.retval,
            self.score,
        ), ()

    def get_args(self):
        return self.args

    def get_choices(self):
        return self.chm

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def project(self, selection: Selection) -> FloatArray:
        weights = list(map(lambda v: v.project(selection), self.chm.submaps))
        return jnp.choose(self.chm.index, weights, mode="wrap")

    def mask_submap(self, concrete_index, argument_index):
        indexed_chm = self.get_choices()
        check = concrete_index == argument_index
        submap = indexed_chm.submaps[concrete_index]
        return BooleanMask.new(check, submap)


#####
# SwitchCombinator
#####


@dataclass
class SwitchCombinator(GenerativeFunction):
    """
    :code:`SwitchCombinator` accepts a set of generative functions as input
    configuration and implements :code:`GenerativeFunction` interface semantics that support branching control flow patterns, including control flow patterns which branch on other stochastic choices.

    This combinator provides a "sum" :code:`Trace` type which allows the internal generative functions to have different choice maps.

    This pattern allows :code:`GenJAX` to express existence
    uncertainty over random choices -- as different generative
    function branches need not share addresses.

    Example
    -------

    .. jupyter-execute::

        import jax
        import genjax
        console = genjax.pretty()

        @genjax.gen
        def branch_1():
            x = genjax.trace("x1", genjax.Normal)(0.0, 1.0)

        @genjax.gen
        def branch_2():
            x = genjax.trace("x2", genjax.Bernoulli)(0.3)

        switch = genjax.SwitchCombinator([branch_1, branch_2])

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(genjax.simulate(switch))
        key, _ = jitted(key, (0, ))
        key, tr = jitted(key, (1, ))
        console.print(tr)
    """

    branches: List[GenerativeFunction]

    def flatten(self):
        return (self.branches,), ()

    def get_trace_type(self, key, args):
        subtypes = []
        for gen_fn in self.branches:
            subtypes.append(gen_fn.get_trace_type(key, args[1:]))
        return SumTraceType(subtypes)

    def _create_sum_pytree(self, key, choices, args):
        covers = []
        for gen_fn in self.branches:
            trace_shape = get_trace_data_shape(gen_fn, key, args)
            covers.append(trace_shape)
        return Sumtree.new(choices, covers)

    def _simulate(self, branch_gen_fn, key, args):
        key, tr = branch_gen_fn.simulate(key, args[1:])
        sum_pytree = self._create_sum_pytree(key, tr, args[1:])
        choices = list(sum_pytree.materialize_iterator())
        branch_index = args[0]
        choice_map = IndexedChoiceMap(branch_index, choices)
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        trace = SwitchTrace(
            self,
            choice_map,
            args,
            retval,
            score,
        )
        return key, trace

    def simulate(self, key, args):
        switch = args[0]

        def _inner(br):
            return lambda key, *args: self._simulate(
                br,
                key,
                args,
            )

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            *args,
        )

    def _importance(self, branch_gen_fn, key, chm, args):
        key, (w, tr) = branch_gen_fn.importance(key, chm, args[1:])
        sum_pytree = self._create_sum_pytree(key, tr, args[1:])
        choices = list(sum_pytree.materialize_iterator())
        branch_index = args[0]
        choice_map = IndexedChoiceMap(branch_index, choices)
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        trace = SwitchTrace(
            self,
            choice_map,
            args,
            retval,
            score,
        )
        return key, (w, trace)

    def importance(self, key, chm, args):
        switch = args[0]

        def _inner(br):
            return lambda key, chm, *args: self._importance(
                br,
                key,
                chm,
                args,
            )

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            chm,
            *args,
        )

    def _update(self, branch_gen_fn, key, prev, new, argdiffs):
        # Create a skeleton discard instance.
        discard_option = BooleanMask.new(False, prev.strip())
        concrete_branch_index = self.branches.index(branch_gen_fn)
        argument_index = strip_diff(argdiffs[0])
        maybe_discard = discard_option.submaps[concrete_branch_index]

        # We have to mask the submap at the concrete_branch_index
        # which we are updating. Why? Because it's possible that the
        # argument_index != concrete_branch_index - meaning we
        # shouldn't perform any inference computations using the submap
        # choices.
        prev = prev.mask_submap(concrete_branch_index, argument_index)

        # Actually perform the update.
        key, (retval_diff, w, tr, actual_discard) = branch_gen_fn.update(
            key,
            prev,
            new,
            argdiffs[1:],
        )

        # Here, we create a Sumtree -- and we place the real trace
        # data inside of it.
        args = jtu.tree_map(strip_diff, argdiffs, is_leaf=check_is_diff)
        sum_pytree = self._create_sum_pytree(key, tr, args[1:])
        choices = list(sum_pytree.materialize_iterator())
        choice_map = IndexedChoiceMap(concrete_branch_index, choices)

        # Merge the skeleton discard with the actual one.
        actual_discard = maybe_discard.merge(actual_discard)
        discard_option.submaps[concrete_branch_index] = actual_discard

        # Get all the metadata for update from the trace.
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        trace = SwitchTrace(
            self,
            choice_map,
            args,
            retval,
            score,
        )
        return key, (retval_diff, w, trace, discard_option)

    def update(self, key, prev, new, argdiffs):
        switch = strip_diff(argdiffs[0])

        def _inner(br):
            return lambda key, prev, new, argdiffs: self._update(
                br, key, prev, new, argdiffs
            )

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(switch, branch_functions, key, prev, new, argdiffs)

    def _assess(self, branch_gen_fn, key, chm, args):
        return branch_gen_fn.assess(key, chm, args[1:])

    def assess(self, key, chm, args):
        switch = args[0]

        def _inner(br):
            return lambda key, chm, *args: self._assess(
                br,
                key,
                chm,
                args,
            )

        branch_functions = list(map(_inner, self.branches))

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            chm,
            *args,
        )
