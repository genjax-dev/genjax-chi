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
from genjax.builtin.trie import Trie
from genjax.core.datatypes import GenerativeFunction, Trace, ChoiceMap
from genjax.builtin.shape_analysis import trace_shape_no_toplevel
from dataclasses import dataclass
from typing import Any, Tuple

#####
# SwitchTrace
#####


@dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    mask: dict
    branch: int
    trie: Trie
    args: Tuple
    retval: Any
    score: jnp.float32

    def get_args(self):
        return self.args

    def get_choices(self):
        return SwitchChoiceMap(self.trie, self.mask)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def flatten(self):
        return (
            self.mask,
            self.branch,
            self.trie,
            self.args,
            self.retval,
            self.score,
        ), (self.gen_fn,)

    @classmethod
    def unflatten(cls, data, xs):
        return SwitchTrace(*data, *xs)


#####
# SwitchChoiceMap
#####


@dataclass
class SwitchChoiceMap(ChoiceMap):
    trie: Trie
    mask: dict

    def has_choice(self, addr):
        if addr not in self.mask:
            return False
        check = self.mask[addr]
        return check

    def get_choice(self, addr):
        if addr not in self.mask:
            raise Exception(f"{addr} not found in SwitchChoiceMap")
        return self.trie[addr]

    def flatten(self):
        return (self.trie, self.mask), ()

    @classmethod
    def unflatten(cls, data, xs):
        return SwitchChoiceMap(*data, *xs)

    def map(self, fn):
        new = Trie({})
        for (k, v) in self.trie:
            new.set_node(k, v.map(fn))
        return SwitchChoiceMap(new, self.mask)

    def get_choices_shallow(self):
        return self.trie.get_choices_shallow()


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
    def compute_branch_coverage_trie(self, key, args):
        trie = Trie({})
        for (_, br) in self.branches.items():
            values, chm_treedef, shape = trace_shape_no_toplevel(br)(key, args)
            # shape = shape.strip_metadata()
            for (k, v) in shape.get_choices_shallow():
                trie.set_node(k, v)
        return trie

    def _simulate(self, branch_gen_fn, key, args):
        emptied = self.compute_branch_coverage_trie(key, args)
        emptied = jtu.tree_map(
            lambda v: jnp.zeros(v.shape, v.dtype),
            emptied,
        )
        branch_index = list(self.branches.values()).index(branch_gen_fn)
        key, tr = branch_gen_fn.simulate(key, args)
        merged = emptied.merge(tr)
        mask = {}
        for (k, _) in merged.get_choices_shallow().items():
            if tr.has_choice(k):
                mask[k] = True
            else:
                mask[k] = False
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        switch_trace = SwitchTrace(
            self, mask, branch_index, merged, args, retval, score
        )
        return key, switch_trace

    def simulate(self, key, args):
        switch = args[0]

        def __inner(br):
            return lambda key, *args: self._simulate(
                br,
                key,
                args,
            )

        branch_functions = list(
            map(
                __inner,
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
        emptied = self.compute_branch_coverage_trie(key, args)
        emptied = jtu.tree_map(
            lambda v: jnp.zeros(v.shape, v.dtype),
            emptied,
        )
        branch_index = list(self.branches.values()).index(branch_gen_fn)
        key, (w, tr) = branch_gen_fn.importance(key, chm, args)
        merged = emptied.merge(tr)
        mask = {}
        for (k, _) in merged.get_choices_shallow().items():
            if tr.has_choice(k):
                mask[k] = True
            else:
                mask[k] = False
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        switch_trace = SwitchTrace(
            self, mask, branch_index, merged, args, retval, score
        )
        return key, (w, switch_trace)

    def importance(self, key, chm, args):
        switch = args[0]

        def __inner(br):
            return lambda key, chm, *args: self._importance(
                br,
                key,
                chm,
                args,
            )

        branch_functions = list(
            map(
                __inner,
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

    def _diff(self, branch_gen_fn, key, prev, new, args):
        key, (w, r) = branch_gen_fn.diff(key, prev, new, args)
        return key, (w, r)

    def diff(self, key, prev, new, args):
        switch = args[0]

        def __inner(br):
            return lambda key, prev, new, *args: self._diff(
                br,
                key,
                prev,
                new,
                args,
            )

        branch_functions = list(
            map(
                __inner,
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

    def _update(self, branch_gen_fn, key, prev, new, args):
        emptied = self.compute_branch_coverage_trie(key, args)
        emptied = jtu.tree_map(
            lambda v: jnp.zeros(v.shape, v.dtype),
            emptied,
        )
        branch_index = list(self.branches.values()).index(branch_gen_fn)
        key, (w, tr, discard) = branch_gen_fn.update(key, prev, new, args)
        merged = emptied.merge(tr)
        mask = {}
        for (k, _) in merged.get_choices_shallow().items():
            if tr.has_choice(k):
                mask[k] = True
            else:
                mask[k] = False
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        switch_trace = SwitchTrace(
            self, mask, branch_index, merged, args, retval, score
        )
        return key, (w, switch_trace, discard)

    def update(self, key, prev, new, args):
        switch = args[0]

        def __inner(br):
            return lambda key, prev, new, *args: self._update(
                br,
                key,
                prev,
                new,
                args,
            )

        branch_functions = list(
            map(
                __inner,
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
