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

import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple
from dataclasses import dataclass
from genjax.builtin.tree import Tree
from genjax.core.datatypes import (
    Trace,
    ChoiceMap,
    Selection,
    GenerativeFunction,
    AllSelection,
    EmptyChoiceMap,
)
from genjax.distributions.distribution import ValueChoiceMap
from genjax.builtin.handlers import (
    handler_sample,
    handler_simulate,
    handler_importance,
    handler_update,
    handler_arg_grad,
    handler_choice_grad,
)
from genjax.builtin.typing import get_trace_type
import genjax.core.pretty_printer as gpp

#####
# Trace
#####


@dataclass
class BuiltinTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Tree
    score: jnp.float32

    def flatten(self):
        return (self.args, self.retval, self.choices, self.score), (
            self.gen_fn,
        )

    @classmethod
    def unflatten(cls, data, xs):
        return BuiltinTrace(*data, *xs)

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return BuiltinChoiceMap(self.choices)

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args


#####
# BuiltinChoiceMap
#####


@dataclass
class BuiltinChoiceMap(ChoiceMap):
    tree: Tree

    def __init__(self, constraints):
        self.tree = Tree({})
        if isinstance(constraints, dict):
            for (k, v) in constraints.items():
                self.tree[k] = (
                    v if isinstance(v, ChoiceMap) else ValueChoiceMap(v)
                )
        elif isinstance(constraints, Tree):
            self.tree = constraints
        elif isinstance(constraints, BuiltinChoiceMap):
            self.tree = constraints.tree

    # Implement the `Pytree` interfaces.
    def flatten(self):
        return (self.tree,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return BuiltinChoiceMap(*xs)

    def overload_pprint(self, **kwargs):
        return gpp._pformat(self.tree, **kwargs)

    def has_choice(self, addr):
        if self.tree.has_node(addr):
            node = self.tree.get_node(addr)
            return not isinstance(node, EmptyChoiceMap)
        else:
            return False

    def get_choice(self, addr):
        if not self.tree.has_node(addr):
            return EmptyChoiceMap()
        node = self.tree.get_node(addr)
        if isinstance(node, Tree):
            return BuiltinChoiceMap(node)
        else:
            return node

    def has_value(self):
        return False

    def get_value(self):
        raise Exception("BuiltinChoiceMap is not a value choice map.")

    def get_choices_shallow(self):
        return self.tree.nodes.items()

    def strip_metadata(self):
        new_tree = Tree({})
        for (k, v) in self.get_choices_shallow():
            new_tree.set_node(k, v.strip_metadata())
        return BuiltinChoiceMap(new_tree)

    def to_selection(self):
        new_tree = Tree({})
        for (k, v) in self.get_choices_shallow():
            new_tree.set_node(k, v.to_selection())
        return BuiltinSelection(new_tree)

    def merge(self, other):
        return BuiltinChoiceMap(self.tree.merge(other))

    def __setitem__(self, k, v):
        self.tree[k] = v

    def __hash__(self):
        return hash(self.tree)


#####
# Selection
#####


@dataclass
class BuiltinSelection(Selection):
    tree: Tree

    def __init__(self, selected):
        self.tree = Tree({})
        if isinstance(selected, list):
            for k in selected:
                self.tree[k] = AllSelection()
        if isinstance(selected, Tree):
            self.tree = selected

    def flatten(self):
        return (self.tree,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return BuiltinSelection(*data, *xs)

    def overload_pprint(self, **kwargs):
        return gpp._pformat(self.tree, **kwargs)

    def filter(self, chm):
        chm = chm.get_choices()

        def _inner(k, v):
            if self.tree.has_node(k):
                sub = self.tree.get_node(k)
                under, s = sub.filter(v)
                return k, under, s
            else:
                return k, EmptyChoiceMap(), 0.0

        new_tree = Tree({})
        score = 0.0
        for (k, v, s) in map(
            lambda args: _inner(*args), chm.get_choices_shallow()
        ):
            if not isinstance(v, EmptyChoiceMap):
                new_tree.set_node(k, v)
                score += s

        return BuiltinChoiceMap(new_tree), score

    def complement(self):
        return BuiltinComplementSelection(self.tree)


@dataclass
class BuiltinComplementSelection(Selection):
    tree: Tree

    def flatten(self):
        return (self.tree,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return BuiltinComplementSelection(*data, *xs)

    def filter(self, chm):
        def _inner(k, v):
            if self.tree.has_node(k):
                sub = self.tree.get_node(k)
                v, s = sub.complement().filter(v)
                return k, v, s
            else:
                return k, v, 0.0

        new_tree = Tree({})
        score = 0.0
        for (k, v, s) in map(
            lambda args: _inner(*args), chm.get_choices_shallow()
        ):
            new_tree.set_node(k, v)
            score += s

        return BuiltinChoiceMap(new_tree), s

    def complement(self):
        new_tree = Tree({})
        for (k, v) in self.tree.get_choices_shallow():
            new_tree[k] = v.complement()
        return BuiltinSelection(new_tree)


#####
# GenerativeFunction
#####


@dataclass
class BuiltinGenerativeFunction(GenerativeFunction):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    @classmethod
    def unflatten(cls, data, xs):
        return BuiltinGenerativeFunction(*data, *xs)

    def __call__(self, key, *args):
        return self.source(key, *args)

    def get_trace_type(self, key, *args, **kwargs):
        jaxpr = jax.make_jaxpr(self.__call__)(key, *args)
        return get_trace_type(jaxpr)

    def sample(self, key, args, **kwargs):
        return handler_sample(self.source, **kwargs)(key, args)

    def simulate(self, key, args, **kwargs):
        key, (f, args, r, chm, score) = handler_simulate(self.source, **kwargs)(
            key, args
        )
        return key, BuiltinTrace(self, args, r, chm, score)

    def importance(self, key, chm, args, **kwargs):
        key, (w, (f, args, r, chm, score)) = handler_importance(
            self.source, **kwargs
        )(key, chm, args)
        return key, (w, BuiltinTrace(self, args, r, chm, score))

    def update(self, key, prev, new, args, **kwargs):
        key, (w, (f, args, r, chm, score), discard) = handler_update(
            self.source, **kwargs
        )(key, prev, new, args)
        return key, (
            w,
            BuiltinTrace(self, args, r, chm, score),
            BuiltinChoiceMap(discard),
        )

    def arg_grad(self, argnums, **kwargs):
        return lambda key, tr, args: handler_arg_grad(
            self.source, argnums, **kwargs
        )(key, tr, args)

    def choice_grad(self, key, tr, selected, **kwargs):
        selected, _ = selected.filter(tr)
        selected = selected.strip_metadata()
        grad_fn = handler_choice_grad(self.source, **kwargs)
        grad, key = jax.grad(
            grad_fn,
            argnums=2,
            has_aux=True,
        )(key, tr, selected)
        return key, grad
