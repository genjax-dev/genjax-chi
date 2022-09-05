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
from .trie import Trie
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

#####
# Trace
#####


@dataclass
class JAXTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    choices: Trie
    score: jnp.float32

    def flatten(self):
        return (self.args, self.retval, self.choices, self.score), (
            self.gen_fn,
        )

    @classmethod
    def unflatten(cls, data, xs):
        return JAXTrace(*data, *xs)

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return JAXChoiceMap(self.choices)

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args


#####
# JAXChoiceMap
#####


@dataclass
class JAXChoiceMap(ChoiceMap):
    trie: Trie

    def __init__(self, constraints):
        self.trie = Trie({})
        if isinstance(constraints, dict):
            for (k, v) in constraints.items():
                self.trie[k] = (
                    v if isinstance(v, ChoiceMap) else ValueChoiceMap(v)
                )
        elif isinstance(constraints, Trie):
            self.trie = constraints
        elif isinstance(constraints, JAXChoiceMap):
            self.trie = constraints.trie

    # Implement the `Pytree` interfaces.
    def flatten(self):
        return (self.trie,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return JAXChoiceMap(*xs)

    def has_choice(self, addr):
        if self.trie.has_node(addr):
            node = self.trie.get_node(addr)
            return not isinstance(node, EmptyChoiceMap)
        else:
            return False

    def get_choice(self, addr):
        if not self.trie.has_node(addr):
            return EmptyChoiceMap()
        node = self.trie.get_node(addr)
        if isinstance(node, Trie):
            return JAXChoiceMap(node)
        else:
            return node

    def has_value(self):
        return False

    def get_value(self):
        raise Exception("JAXChoiceMap is not a value choice map.")

    def get_choices_shallow(self):
        return self.trie.nodes.items()

    def strip_metadata(self):
        new_trie = Trie({})
        for (k, v) in self.get_choices_shallow():
            new_trie.set_node(k, v.strip_metadata())
        return JAXChoiceMap(new_trie)

    def to_selection(self):
        new_trie = Trie({})
        for (k, v) in self.get_choices_shallow():
            new_trie.set_node(k, v.to_selection())
        return JAXSelection(new_trie)

    def merge(self, other):
        return JAXChoiceMap(self.trie.merge(other))

    def __setitem__(self, k, v):
        self.trie[k] = v


#####
# Selection
#####


@dataclass
class JAXSelection(Selection):
    trie: Trie

    def __init__(self, selected):
        self.trie = Trie({})
        if isinstance(selected, list):
            for k in selected:
                self.trie[k] = AllSelection()
        if isinstance(selected, Trie):
            self.trie = selected

    def flatten(self):
        return (self.trie,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return JAXSelection(*data, *xs)

    def filter(self, chm):
        chm = chm.get_choices()

        def _inner(k, v):
            if self.trie.has_node(k):
                sub = self.trie.get_node(k)
                return k, sub.filter(v)
            else:
                return k, EmptyChoiceMap()

        new_trie = Trie({})
        for k, v in map(lambda args: _inner(*args), chm.get_choices_shallow()):
            new_trie.set_node(k, v)

        return JAXChoiceMap(new_trie)

    def complement(self):
        return JAXComplementSelection(self.trie)


@dataclass
class JAXComplementSelection(Selection):
    trie: Trie

    def flatten(self):
        return (self.trie,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return JAXComplementSelection(*data, *xs)

    def filter(self, chm):
        def _inner(k, v):
            if self.trie.has_node(k):
                sub = self.trie.get_node(k)
                return k, sub.complement().filter(v)
            else:
                return k, v

        new_trie = Trie({})
        for k, v in map(lambda args: _inner(*args), chm.get_choices_shallow()):
            new_trie.set_node(k, v)

        return JAXChoiceMap(new_trie)

    def complement(self):
        new_trie = Trie({})
        for (k, v) in self.trie.get_choices_shallow():
            new_trie[k] = v.complement()
        return JAXSelection(new_trie)


#####
# GenerativeFunction
#####


@dataclass
class JAXGenerativeFunction(GenerativeFunction):
    source: Callable

    def __call__(self, key, *args):
        return self.source(key, *args)

    def flatten(self):
        return (), (self.source,)

    @classmethod
    def unflatten(cls, data, xs):
        return JAXGenerativeFunction(*data, *xs)

    def sample(self, key, args, **kwargs):
        return handler_sample(self.source, **kwargs)(key, args)

    def simulate(self, key, args, **kwargs):
        key, (f, args, r, chm, score) = handler_simulate(self.source, **kwargs)(
            key, args
        )
        return key, JAXTrace(self, args, r, chm, score)

    def importance(self, key, chm, args, **kwargs):
        key, (w, (f, args, r, chm, score)) = handler_importance(
            self.source, **kwargs
        )(key, chm, args)
        return key, (w, JAXTrace(self, args, r, chm, score))

    def update(self, key, prev, new, args, **kwargs):
        key, (w, (f, args, r, chm, score), discard) = handler_update(
            self.source, **kwargs
        )(key, prev, new, args)
        return key, (
            w,
            JAXTrace(self, args, r, chm, score),
            JAXChoiceMap(discard),
        )

    def arg_grad(self, argnums, **kwargs):
        return lambda key, tr, args: handler_arg_grad(
            self.source, argnums, **kwargs
        )(key, tr, args)

    def choice_grad(self, key, tr, selected, **kwargs):
        selected = selected.filter(tr).strip_metadata()
        grad_fn = handler_choice_grad(self.source, **kwargs)
        grad, key = jax.grad(
            grad_fn,
            argnums=2,
            has_aux=True,
        )(key, tr, selected)
        return key, grad
