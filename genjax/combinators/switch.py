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
structured control flow for combinations of generative functions
which can return different shaped choice maps.

It's based on encoding a "sum type" like trace using JAX.

Generative functions which are passed in as branches to `SwitchCombinator`
must accept the same argument types, and return the same type of return value.

The internal choice maps for the branch generative functions
can have different shape/dtype choices.
"""

import jax
import jax.numpy as jnp
from genjax.core.datatypes import GenerativeFunction, Trace
from genjax.builtin.shape_analysis import abstract_choice_map_shape
from dataclasses import dataclass
from typing import Tuple, Any, Sequence

#####
# SwitchTrace
#####


@dataclass
class SwitchTrace(Trace):
    gen_fn: GenerativeFunction
    forms: dict
    payload_tree: dict
    branch: int
    args: Tuple
    retval: Any
    score: jnp.float32

    def get_args(self):
        return self.args

    def get_choices(self):
        return self.payload_tree

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def flatten(self):
        return (
            self.payload_tree,
            self.branch,
            self.args,
            self.retval,
            self.score,
        ), (
            self.gen_fn,
            self.forms,
        )

    @classmethod
    def unflatten(cls, data, xs):
        return SwitchTrace(*data, *xs)


#####
# SwitchCombinator
#####


@dataclass
class SwitchCombinator(GenerativeFunction):
    branches: Sequence[GenerativeFunction]

    def __init__(self, *branches):
        self.branches = branches

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
    def compute_branch_coverage(self, key, args):
        forms = {}
        shaped_structs = set()
        branch_maps = []
        for (ind, br) in enumerate(self.branches):
            values, trace_treedef = abstract_choice_map_shape(br)(key, args)
            forms[ind] = trace_treedef
            shaped_structs.update(set(values))
            local_shapes = {}
            for v in values:
                num = local_shapes.get((v.shape, v.dtype), 0)
                local_shapes[(v.shape, v.dtype)] = num + 1
            branch_maps.append(local_shapes)
        coverage = {}
        for shape in shaped_structs:
            coverage_num = 0
            for br_map in branch_maps:
                branch_num = br_map.get((shape.shape, shape.dtype), 0)
                if branch_num > coverage_num:
                    coverage_num = branch_num
            coverage[(shape.shape, shape.dtype)] = coverage_num
        return coverage, forms

    def _simulate(self, switch, branch_gen_fn, key, args, forms, coverage):
        key, tr = branch_gen_fn.simulate(key, args)
        payload = {}
        leaves = map(
            lambda l: (l.shape, l.dtype), jax.tree_util.tree_leaves(tr)
        )
        for leaf in leaves:
            payload_tree = payload.get(leaf, [])
            payload_tree.append(leaves)

        for ((shape, dtype), v) in coverage.items():
            payload_tree = payload.get((shape, dtype), [])
            if len(payload_tree) < v:
                payload_tree.append(
                    [
                        jnp.zeros(shape, dtype=dtype)
                        for _ in range(0, v - len(payload_tree))
                    ]
                )
        score = tr.get_score()
        args = tr.get_args()
        retval = tr.get_retval()
        switch_trace = SwitchTrace(
            self, forms, payload_tree, switch, args, retval, score
        )
        return key, switch_trace

    def simulate(self, key, args):
        switch = args[0]
        coverage, forms = self.compute_branch_coverage(key, args[1:])

        def __inner(br):
            return lambda switch, key, *args: self._simulate(
                switch, br, key, args, forms, coverage
            )

        branch_functions = list(
            map(
                __inner,
                self.branches,
            )
        )

        return jax.lax.switch(
            switch,
            branch_functions,
            switch,
            key,
            *args[1:],
        )
