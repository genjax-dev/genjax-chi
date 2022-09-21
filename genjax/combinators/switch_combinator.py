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

"""
This module implements a generative function combinator which allows
branching control flow for combinations of generative functions
which can return different shaped choice maps.
It's based on encoding a trace sum type using JAX - to bypass restrictions from `jax.lax.switch`_.

Generative functions which are passed in as branches to :code:`SwitchCombinator`
must accept the same argument types, and return the same type of return value.

The internal choice maps for the branch generative functions
can have different shape/dtype choices.

.. _jax.lax.switch: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.switch.html
"""

import jax
import jax.numpy as jnp
from genjax.core.pytree import SumPytree
from genjax.core.datatypes import GenerativeFunction, Trace
from genjax.core.masks import BooleanMask
from genjax.combinators.combinator_datatypes import IndexedChoiceMap
from genjax.combinators.combinator_tracetypes import SumTraceType
from dataclasses import dataclass
from typing import Any, Tuple, Sequence

#####
# SwitchTrace
#####


@dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    chm: IndexedChoiceMap
    args: Tuple
    retval: Any
    score: jnp.float32

    def flatten(self):
        return (
            self.chm,
            self.args,
            self.retval,
            self.score,
        ), (self.gen_fn,)

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

    def convert_to_boolean_mask(self, passed_in_index, argument_index):
        indexed_chm = self.get_choices()
        check = passed_in_index == argument_index
        submap = indexed_chm.submaps[passed_in_index]
        return BooleanMask.new(
            check,
            submap,
        )


#####
# SwitchCombinator
#####


@dataclass
class SwitchCombinator(GenerativeFunction):
    """
    :code:`SwitchCombinator` accepts a set of generative functions as input
    configuration and implements a branching control flow pattern. This
    combinator provides a "sum" :code:`Trace` type which allows the internal
    generative functions to have different choice maps.

    This pattern allows :code:`GenJAX` to express existence uncertainty
    over random choices -- as different generative function branches
    need not share addresses.

    Usage of the :doc:`interface` is detailed below under each
    method implementation.

    Parameters
    ----------

    *args: :code:`GenerativeFunction`
        A splatted sequence of `GenerativeFunction` instances should be provided
        to the :code:`SwitchCombinator` constructor.

    Returns
    -------

    :code:`SwitchCombinator`
        A single :code:`SwitchCombinator` generative function which
        implements a branch control flow pattern using each
        provided internal generative function (see parameters) as
        a potential branch.

    Example
    -------

    .. jupyter-execute::

        import jax
        import genjax

        @genjax.gen
        def branch_1(key):
            key, x = genjax.trace("x1", genjax.Normal)(key, (0.0, 1.0))
            return (key, )

        @genjax.gen
        def branch_2(key):
            key, x = genjax.trace("x2", genjax.Bernoulli)(key, (0.3, ))
            return (key, )

        switch = genjax.SwitchCombinator([branch_1, branch_2])

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(genjax.simulate(switch))
        key, _ = jitted(key, (0, ))
        key, tr = jitted(key, (1, ))
        print(tr)
    """

    branches: dict[int, GenerativeFunction]

    def __init__(self, branches: Sequence):
        self.branches = {}
        for (ind, gen_fn) in enumerate(branches):
            self.branches[ind] = gen_fn

    def flatten(self):
        return (), (self.branches,)

    def __call__(self, key, *args):
        return jax.lax.switch(
            args[0],
            self.branches,
            key,
            *args[1:],
        )

    def get_trace_type(self, key, args):
        subtypes = []
        for (_, gen_fn) in self.branches.items():
            subtypes.append(gen_fn.get_trace_type(key, args[1:]))
        return SumTraceType(subtypes)

    def create_sum_pytree(self, key, choices, args):
        covers = []
        for (_, gen_fn) in self.branches.items():
            _, (key, abstr) = jax.make_jaxpr(
                gen_fn.simulate, return_shape=True
            )(key, args)
            covers.append(abstr)
        return SumPytree.new(choices, covers)

    def _simulate(self, branch_gen_fn, key, args):
        key, tr = branch_gen_fn.simulate(key, args[1:])
        choices = tr.get_choices()
        sum_pytree = self.create_sum_pytree(key, choices, args[1:])
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

        branch_functions = list(
            map(
                _inner,
                self.branches.values(),
            )
        )

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            *args,
        )

    def _importance(self, branch_gen_fn, key, chm, args):
        key, (w, tr) = branch_gen_fn.importance(key, chm, args[1:])
        choices = tr.get_choices()
        sum_pytree = self.create_sum_pytree(key, choices, args[1:])
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

    @IndexedChoiceMap.collapse_boundary
    @BooleanMask.collapse_boundary
    def importance(self, key, chm, args):
        switch = args[0]

        def _inner(br):
            return lambda key, chm, *args: self._importance(
                br,
                key,
                chm,
                args,
            )

        branch_functions = list(
            map(
                _inner,
                self.branches.values(),
            )
        )

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            chm,
            *args,
        )

    def _update(self, branch_gen_fn, key, prev, new, args):
        discard_option = BooleanMask.new(
            True, prev.strip_metadata()
        ).leaf_push()
        concrete_branch_index = list(self.branches.keys())[
            list(self.branches.values()).index(branch_gen_fn)
        ]
        argument_index = args[0]
        prev = prev.convert_to_boolean_mask(
            concrete_branch_index, argument_index
        )
        discard_branch = discard_option.submaps[concrete_branch_index]
        discard_branch = BooleanMask.new(False, discard_branch).leaf_push()
        key, (w, tr, discard) = branch_gen_fn.update(key, prev, new, args[1:])
        choices = tr.get_choices()
        sum_pytree = self.create_sum_pytree(key, choices, args[1:])
        choices = list(sum_pytree.materialize_iterator())
        choice_map = IndexedChoiceMap(concrete_branch_index, choices)
        discard_branch = discard_branch.merge(discard)
        discard_option.submaps[concrete_branch_index] = discard_branch
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
        return key, (w, trace, discard_option)

    @IndexedChoiceMap.collapse_boundary
    @BooleanMask.collapse_boundary
    def update(self, key, prev, new, args):
        switch = args[0]

        def _inner(br):
            return lambda key, prev, new, *args: self._update(
                br,
                key,
                prev,
                new,
                args,
            )

        branch_functions = list(
            map(
                _inner,
                self.branches.values(),
            )
        )

        return jax.lax.switch(
            switch,
            branch_functions,
            key,
            prev,
            new,
            *args,
        )
