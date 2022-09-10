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
It's based on encoding a trace sum type using JAX - to bypass restrictions from `jax.lax.cond`_.

Generative functions which are passed in as branches to :code:`SwitchCombinator`
must accept the same argument types, and return the same type of return value.

The internal choice maps for the branch generative functions
can have different shape/dtype choices.

.. _jax.lax.cond: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from genjax.core.datatypes import (
    GenerativeFunction,
    Trace,
    BooleanMask,
)
from genjax.distributions.distribution import ValueChoiceMap
from genjax.builtin.shape_analysis import trace_shape_no_toplevel
from genjax.builtin.tree import Tree
from dataclasses import dataclass
from typing import Any, Tuple

#####
# SwitchTrace
#####


@dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    chm: BooleanMask
    args: Tuple
    retval: Any
    score: jnp.float32

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

    def flatten(self):
        return (
            self.chm,
            self.args,
            self.retval,
            self.score,
        ), (self.gen_fn,)

    @classmethod
    def unflatten(cls, data, xs):
        return SwitchTrace(*data, *xs)


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

        switch = genjax.SwitchCombinator(branch_1, branch_2)

        key = jax.random.PRNGKey(314159)
        jitted = jax.jit(genjax.simulate(switch))
        key, _ = jitted(key, (0, ))
        key, tr = jitted(key, (1, ))
        print(tr)
    """

    branches: dict[int, GenerativeFunction]

    def __init__(self, *branches):
        self.branches = {}
        for (ind, gen_fn) in enumerate(branches):
            self.branches[ind] = gen_fn

    def flatten(self):
        return (), (self.branches,)

    @classmethod
    def unflatten(cls, xs, data):
        return SwitchCombinator(*xs, *data)

    def __call__(self, key, *args):
        return jax.lax.switch(
            args[0],
            self.branches,
            key,
            *args[1:],
        )

    # This function does some compile-time code specialization
    # to produce a "sum type" - like trace.
    def create_masked_tree(self, key, args):
        tree = Tree({})
        for (_, br) in self.branches.items():
            _, _, shape = trace_shape_no_toplevel(br)(key, args)
            shape = jtu.tree_map(
                lambda v: jnp.zeros(v.shape, v.dtype),
                shape,
                is_leaf=lambda v: isinstance(v, ValueChoiceMap),
            )
            tree = tree.merge(shape)
        return BooleanMask(tree, False)

    def _simulate(self, branch_gen_fn, key, args):
        tree = self.create_masked_tree(key, args)
        key, tr = branch_gen_fn.simulate(key, args)
        choices = BooleanMask(tr.get_choices(), True)
        tree = tree.merge(choices)
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        trace = SwitchTrace(self, tree, args, retval, score)
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
            *args[1:],
        )

    def _importance(self, branch_gen_fn, key, chm, args):
        tree = self.create_masked_tree(key, args)
        key, (w, tr) = branch_gen_fn.importance(key, chm, args)
        choices = BooleanMask(tr.get_choices(), True)
        tree = tree.merge(choices)
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        trace = SwitchTrace(self, tree, args, retval, score)
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
            *args[1:],
        )

    def _update(self, branch_gen_fn, key, prev, discard_option, new, args):
        tree = self.create_masked_tree(key, args)
        key, (w, tr, discard) = branch_gen_fn.update(key, prev, new, args)
        discard = discard_option.merge(discard)
        choices = BooleanMask(tr.get_choices(), True)
        tree = tree.merge(choices)
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        trace = SwitchTrace(self, tree, args, retval, score)
        return key, (w, trace, discard)

    def update(self, key, prev, new, args):
        switch = args[0]
        discard_option = BooleanMask(prev.get_choices(), False)

        def _inner(br):
            return lambda key, prev, new, *args: self._update(
                br,
                key,
                prev,
                discard_option,
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
            *args[1:],
        )
